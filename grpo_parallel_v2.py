#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO State Tuning - Parallel Inference Mode
- Model: rwkv7-g1c-2.9b-20251231-ctx8192
- batch_prompts=8, group_size=8 (8 questions * 8 parallel samples = 64 outputs)
- Uses optimized FP16 inference framework (rwkv7_fp16 with v2 kernel)
- Optimizer: Adam (no weight decay), eps=1e-18, lr=1e-4
"""

import os
import sys




import re
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def read_parquet(path: str) -> List[Dict[str, Any]]:
    """Read parquet file and convert to list of dicts with 'problem' and 'solution' keys"""
    import pandas as pd
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        # Map 'question' -> 'problem', 'answer' -> 'solution'
        data.append({
            "problem": row.get("question", row.get("problem", "")),
            "solution": row.get("answer", row.get("solution", ""))
        })
    return data

def read_json(path: str) -> List[Dict[str, Any]]:
    """Read json file (list format) and convert to list of dicts with 'problem' and 'solution' keys"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = []
    for item in raw:
        data.append({
            "problem": item.get("question", item.get("problem", "")),
            "solution": item.get("answer", item.get("solution", ""))
        })
    return data

def load_data(path: str) -> List[Dict[str, Any]]:
    """Load data from jsonl, parquet, or json file"""
    if path.endswith(".parquet"):
        return read_parquet(path)
    elif path.endswith(".json"):
        return read_json(path)
    else:
        return read_jsonl(path)

def append_jsonl(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# Prompt (keep 'think'!)
# =========================================================

def build_prompt(problem: str) -> str:
    p = (problem or "").strip()
    return (
        f"User: {p}\n"
        f"请将最终答案放在\\boxed{{...}}里，并且最终只给出\\boxed{{...}}这一行，不要输出多余内容。 think\n"
        f"Assistant: <think>\n"
    )


# =========================================================
# Answer extraction & judging
# =========================================================

def _strip_math_delims(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,\;\!\:]\s*", "", s)
    return s.strip()

def _find_balanced_brace(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    if brace_start < 0 or brace_start >= len(text) or text[brace_start] != "{":
        return None
    depth = 0
    i = brace_start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i], i
        i += 1
    return None

def extract_last_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    brace = idx + len(key) - 1
    got = _find_balanced_brace(text, brace)
    if got is None:
        return None
    inner, _ = got
    return _strip_math_delims(inner)

def extract_final_answer(text: str) -> Optional[str]:
    a = extract_last_boxed(text)
    if a:
        return a
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if not lines:
        return None
    last = lines[-1].replace("</think>", "").strip()
    return _strip_math_delims(last) if last else None

def boxed_complete(text: str) -> bool:
    k = text.find(r"\boxed{")
    if k < 0:
        return False
    i = k + len(r"\boxed{")
    start = i
    depth = 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # Check if content is placeholder like "..."
                content = text[start:i].strip()
                if content in ('...', '…', '．．．', ''):
                    return False
                return True
        i += 1
    return False

def _latex_to_sympyish(s: str) -> str:
    if s is None:
        return ""
    s = _strip_math_delims(s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace("^", "**")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\infty", "oo").replace("∞", "oo")
    s = s.replace("−", "-")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    while True:
        idx = s.find(r"\frac{")
        if idx < 0:
            break
        brace1 = idx + len(r"\frac")
        got1 = _find_balanced_brace(s, brace1)
        if got1 is None:
            break
        a, end1 = got1
        if end1 + 1 >= len(s) or s[end1 + 1] != "{":
            break
        got2 = _find_balanced_brace(s, end1 + 1)
        if got2 is None:
            break
        b, end2 = got2
        s = s[:idx] + f"(({_latex_to_sympyish(a)})/({_latex_to_sympyish(b)}))" + s[end2 + 1:]

    while True:
        idx = s.find(r"\sqrt{")
        if idx < 0:
            break
        brace = idx + len(r"\sqrt")
        got = _find_balanced_brace(s, brace)
        if got is None:
            break
        inner, end = got
        s = s[:idx] + f"sqrt({_latex_to_sympyish(inner)})" + s[end + 1:]

    s = s.replace("\\", "")
    return s.strip()

# =========================================================
# Zhipu API judge (via OpenAI-compatible API)
# =========================================================

import requests

ZHIPU_API_URL = ""
ZHIPU_API_KEY = ""
ZHIPU_MODEL = ""


def judge_with_zhipu(pred_full_output: str, gt: str) -> Tuple[bool, str]:
    pred = (pred_full_output or "").strip()
    gt = (gt or "").strip()

    prompt = (
        "你是一个严格的答案判定器。给定标准答案(GT)与模型的完整输出(OUTPUT)，判断模型的回答是否正确。\n"
        "你需要从模型输出中找到最终答案（通常在\\boxed{}中），然后判断其是否与标准答案等价。\n"
        "等价包括：数学等价、同义表述、可化简的表达式、等值分数/小数等。\n"
        "若无法确定或信息不足，一律判为不等价并输出 0。\n"
        "请只输出一个字符：1 或 0。\n\n"
        f"GT: {gt}\n"
        f"OUTPUT: {pred}\n"
        "Output: "
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_KEY}"
    }

    data = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    for attempt in range(3):
        try:
            response = requests.post(ZHIPU_API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # If raw is empty or doesn't contain 0/1, raise exception to trigger fallback
            if not raw or not raw.strip():
                raise ValueError("Empty API response")

            m = re.search(r"(?<!\d)([01])(?!\d)", raw.strip())
            if m is None:
                raise ValueError(f"Invalid API response: {raw[:50]}")

            ok = (m.group(1) == "1")
            return ok, raw
        except Exception as e:
            if attempt < 2:
                time.sleep(0.5)
                continue
            raise e


def extract_gsmk_gt(answer_text: str) -> str:
    """Extract final answer from GSM8K-like rationale string: ... #### 70"""
    if answer_text is None:
        return ""
    m = re.search(r"####\s*(.*)", str(answer_text))
    if m:
        tail = m.group(1).strip()
        for line in tail.splitlines():
            line = line.strip()
            if line:
                return line
        return tail.strip()
    lines = [x.strip() for x in str(answer_text).splitlines() if x.strip()]
    return lines[-1] if lines else ""


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}"""
    if not text:
        return None
    # Find all \boxed{...}
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    # Filter out placeholder patterns like "...", "…", empty
    valid_matches = [m.strip() for m in matches if m.strip() and m.strip() not in ('...', '…', '．．．')]
    if valid_matches:
        return valid_matches[-1]
    return None


def judge_answer(pred_text: str, gt_text: str) -> Tuple[float, Dict[str, Any]]:
    """Local judge - check if GT integer appears in boxed answer or output"""
    # Extract GT integer (from "#### 70" format)
    gt_ans = extract_gsmk_gt(gt_text)

    dbg: Dict[str, Any] = {
        "pred_full_output": pred_text[:500] if pred_text else None,
        "gt": gt_ans,
        "method": "local",
        "error": None,
        "boxed_answer": None,
        "correct": False,
    }

    if not pred_text or not pred_text.strip():
        dbg["method"] = "no_output"
        return 0.0, dbg

    # First try to extract from \boxed{}
    boxed = extract_boxed_answer(pred_text)
    if boxed:
        dbg["boxed_answer"] = boxed
        dbg["method"] = "boxed"
        ok = (gt_ans in boxed) or (boxed == gt_ans)
        dbg["correct"] = ok
        return (1.0 if ok else 0.0), dbg

    # Fallback: check if GT appears anywhere in output
    dbg["method"] = "fallback_contains"
    ok = (gt_ans in pred_text)
    dbg["correct"] = ok
    return (1.0 if ok else 0.0), dbg


# =========================================================
# Config - Modified for GRPO
# =========================================================

@dataclass
class GRPOConfig:
    # sampling / batch - 8 questions * 16 parallel samples = 128 outputs
    batch_prompts: int = 8
    group_size: int = 8
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20  # -1 for no limit with Rapid-Sampling
    mask_token0: bool = True
    decay: float = 0.99

    # Rapid-Sampling repetition penalty
    use_rapid_sampling: bool = True
    presence_penalty: float = 0.8
    repetition_penalty: float = 0.6
    penalty_decay: float = 0.995

    # stop checks
    stop_on_think_close: bool = False  # Disabled: let model generate \boxed{} after </think>
    stop_on_user: bool = True
    stop_on_boxed: bool = True
    stop_check_every: int = 8
    stop_check_window: int = 96

    # dynamic sampling - disabled for GRPO (no filtering)
    dynamic_sampling_max_tries: int = 1  # Only 1 try since we don't filter
    collect_chunk: int = 8  # Same as batch_prompts
    collect_rounds: int = 1  # Number of sampling rounds per step (for accumulating more samples)
    min_valid_groups: int = 4  # 最少有效组数（非all0/all1），达到后停止rollout
    max_rollout_rounds: int = 10  # 最大rollout轮数，防止无限循环

    # PPO/GRPO
    ppo_epochs: int = 2
    micro_batch: int = 4
    lr: float = 1e-4
    eps_low: float = 0.2
    eps_high: float = 0.5
    grad_clip: float = 0.2

    # stability
    kl_coef: float = 0.001
    target_kl: float = 0.01
    adaptive_kl: bool = True
    kl_early_stop_factor: float = 1.5  # PPO epoch提前停止：当kl > target_kl * factor时
    time_state_l2: float = 1e-6
    time_state_clamp: float = 0.0  # Disabled

    # logging / save
    log_interval: int = 1  # More frequent logging
    save_interval: int = 20
    infer_check_interval: int = 50

    # eval
    eval_interval: int = 20
    eval_n: int = 128
    eval_temperature: float = 0.3
    eval_top_p: float = 0.4
    eval_top_k: int = 500
    eval_max_new_tokens: int = 2048
    eval_presence_penalty: float = 0.5
    eval_frequency_penalty: float = 0.1
    eval_penalty_decay: float = 0.99

    # faulthandler
    enable_faulthandler: bool = False
    hang_dump_s: float = 0.0


# =========================================================
# Model helpers
# =========================================================

HEAD_SIZE = 64

def normalize_model_arg(model_arg: str) -> Tuple[str, str]:
    model_arg = model_arg.strip()
    if model_arg.endswith(".pth"):
        base = model_arg[:-4]
        pth = model_arg
    else:
        base = model_arg
        pth = model_arg + ".pth"
    if not os.path.isfile(pth) and os.path.isfile(base):
        pth = model_arg
        if pth.endswith(".pth"):
            base = pth[:-4]
    return base, pth

def _torch_load_weights(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_train_model_rwkv7_cuda(pth_path: str, device: str, ctx_len: int, grad_cp: int = 0):
    from types import SimpleNamespace
    from rwkv7_trainable import RWKV7

    sd = _torch_load_weights(pth_path)

    n_embd = sd["emb.weight"].shape[1]
    vocab_size = sd["emb.weight"].shape[0]
    n_layer = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    dim_ffn = sd.get("blocks.0.ffn.key.weight", torch.zeros(n_embd * 4, n_embd)).shape[0]

    args = SimpleNamespace(
        n_embd=n_embd,
        vocab_size=vocab_size,
        n_layer=n_layer,
        dim_att=n_embd,
        dim_ffn=dim_ffn,
        head_size_a=HEAD_SIZE,
        head_size_divisor=8,
        ctx_len=ctx_len,
        chunk_ctx=ctx_len,
        grad_cp=grad_cp,
        train_type="state",
        peft="none",
        my_testing="x070",
    )

    model = RWKV7(args)
    model.load_state_dict(sd, strict=False)
    model.args = args
    model = model.to(device).to(torch.bfloat16)
    return model, args

def load_infer_model_fp16(base_name_no_pth: str, device: str = "cuda"):
    """Load optimized FP16 inference model with v2 kernel"""
    import types

    from rwkv7_fp16 import RWKV_x070

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.MODEL_NAME = base_name_no_pth
    model = RWKV_x070(args)

    return model, args

def freeze_except_time_state(model: torch.nn.Module) -> int:
    cnt = 0
    for n, p in model.named_parameters():
        if "time_state" in n:
            p.requires_grad = True
            cnt += p.numel()
        else:
            p.requires_grad = False
    return cnt

def save_time_state_only(model: torch.nn.Module, path: str):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if "time_state" in n}
    torch.save(sd, path)

def load_time_state_only(model: torch.nn.Module, path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    sd = _torch_load_weights(path)
    if 'time_state' in sd and isinstance(sd['time_state'], dict):
        sd = sd['time_state']
    hit = 0
    for n, p in model.named_parameters():
        if n in sd:
            p.data.copy_(sd[n].to(p.device).to(p.dtype))
            hit += 1
    return hit > 0


# =========================================================
# FP16 batched inference (parallel group sampling)
# =========================================================

class FP16BatchInference:
    def __init__(self, infer_model, train_model, encode_fn, decode_fn, device: str, cfg: GRPOConfig):
        self.infer_model = infer_model
        self.train_model = train_model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.cfg = cfg
        self.vocab_size = infer_model.args.vocab_size

        # Load Rapid-Sampling kernel if enabled
        self.sample_kernel = None
        if cfg.use_rapid_sampling:
            from torch.utils.cpp_extension import load
            print("Loading Rapid-Sampling kernel...")
            self.sample_kernel = load(
                name="sample",
                sources=[f"{RAPID_SAMPLING_DIR}/sampling.cpp", f"{RAPID_SAMPLING_DIR}/sampling.cu"],
                extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
                verbose=False,
            )
            print("Rapid-Sampling kernel loaded.")

    def init_state_with_time_state(self, B: int):
        # Use the device passed to this class
        infer_device = self.device
        args = self.infer_model.args

        # Create state tensors on the correct device
        state = [None, None]
        DTYPE = torch.half  # fp16
        state[0] = torch.zeros((args.n_layer, 2, B, args.n_embd), dtype=DTYPE, requires_grad=False, device=infer_device)
        state[1] = torch.zeros((args.n_layer, B, args.n_embd // args.head_size, args.head_size, args.head_size),
                               dtype=DTYPE, requires_grad=False, device=infer_device)

        # Load trained time_state
        for i, block in enumerate(self.train_model.blocks):
            ts = block.att.time_state  # (H,64,64) - bfloat16 on train device
            # Convert bfloat16 -> fp16 and move to infer device
            state[1][i] = ts.unsqueeze(0).expand(B, -1, -1, -1).clone().to(device=infer_device, dtype=torch.half)
        return state

    @torch.no_grad()
    def prime_prompts(self, prompt_tokens_list: List[List[int]]):
        B = len(prompt_tokens_list)
        state = self.init_state_with_time_state(B)
        out = self.infer_model.forward_batch(prompt_tokens_list, state)
        if torch.is_tensor(out) and out.dim() == 3:
            out = out[:, -1, :]
        return out, state

    @torch.no_grad()
    def generate_group_parallel(
        self,
        prompt_tokens_list: List[List[int]],
        group_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_on_think_close: bool,
        stop_on_user: bool,
        stop_on_boxed: bool,
        stop_check_every: int,
        stop_check_window: int,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        penalty_decay: float = None,
    ) -> Tuple[List[List[int]], List[List[float]], List[str], List[bool]]:

        # Use passed params or fall back to cfg defaults
        presence_penalty = presence_penalty if presence_penalty is not None else self.cfg.presence_penalty
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.cfg.repetition_penalty
        penalty_decay = penalty_decay if penalty_decay is not None else self.cfg.penalty_decay

        Bp = len(prompt_tokens_list)
        if Bp == 0:
            return [], [], [], []

        last_logits, state = self.prime_prompts(prompt_tokens_list)

        B = Bp * group_size
        last_logits = last_logits.repeat_interleave(group_size, dim=0).contiguous()

        # repeat state for group
        state0 = state[0].repeat_interleave(group_size, dim=2).contiguous()
        state1 = state[1].repeat_interleave(group_size, dim=1).contiguous()
        state = [state0, state1]

        comp_tokens: List[List[int]] = [[] for _ in range(B)]
        old_logps: List[List[float]] = [[] for _ in range(B)]
        active = torch.ones((B,), device=last_logits.device, dtype=torch.bool)
        truncated = [False for _ in range(B)]

        # Setup for Rapid-Sampling
        if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
            rand_states = self.sample_kernel.setup_rand(int(time.time()), B)
            # vocab_size needs to be multiple of 4
            vocab_padded = ((self.vocab_size + 3) // 4) * 4
            penalties = torch.zeros(B, vocab_padded, device=self.device, dtype=torch.float32)

        def sample_next(logits_2d: torch.Tensor) -> torch.Tensor:
            # Use Rapid-Sampling kernel if available
            if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
                logits_float = logits_2d.float()
                # Pad logits to multiple of 4
                if logits_float.size(-1) % 4 != 0:
                    pad_size = 4 - (logits_float.size(-1) % 4)
                    logits_float = F.pad(logits_float, (0, pad_size), value=-1e30)

                return self.sample_kernel.batch_sampling_repetition_temperature_topk_topp(
                    logits_float,
                    penalties,
                    rand_states,
                    presence_penalty,
                    frequency_penalty,
                    penalty_decay,
                    temperature,
                    top_k,
                    top_p
                )

            # Fallback to original sampling
            if temperature <= 0:
                return torch.argmax(logits_2d, dim=-1)

            x = logits_2d.float() / float(temperature)
            V = x.size(-1)

            if self.cfg.mask_token0:
                x[:, 0] = -1e30

            k_cap = 0
            if top_k and top_k > 0:
                k_cap = int(min(top_k, V))
            elif top_p and 0.0 < top_p < 1.0:
                k_cap = int(min(2048, V))

            if k_cap > 0:
                topv, topi = torch.topk(x, k=k_cap, dim=-1)
                if top_p and 0.0 < top_p < 1.0:
                    probs = F.softmax(topv, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1)
                    keep = cdf <= float(top_p)
                    keep[:, 0] = True
                    topv = topv.masked_fill(~keep, -1e30)
                probs = F.softmax(topv, dim=-1)
                pick = torch.multinomial(probs, 1).squeeze(-1)
                return topi.gather(-1, pick.unsqueeze(-1)).squeeze(-1)

            probs = F.softmax(x, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

        for t in range(max_new_tokens):
            if not bool(active.any().item()):
                break

            logits = last_logits
            token_ids = sample_next(logits).long()  # Ensure int64 for gather

            logp_all = F.log_softmax(logits.float(), dim=-1)
            picked_logp = logp_all.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

            token_ids = torch.where(active, token_ids, torch.zeros_like(token_ids))
            picked_logp = torch.where(active, picked_logp, torch.zeros_like(picked_logp))

            tok_cpu = token_ids.detach().cpu().tolist()
            lp_cpu = picked_logp.detach().cpu().tolist()

            for i in range(B):
                if not active[i]:
                    continue
                comp_tokens[i].append(int(tok_cpu[i]))
                old_logps[i].append(float(lp_cpu[i]))

            # stop token check (0, 261, 24281)
            STOP_TOKENS = {0, 261, 24281}
            for i in range(B):
                if active[i] and int(tok_cpu[i]) in STOP_TOKENS:
                    active[i] = False

            # stop check
            if (stop_on_think_close or stop_on_user or stop_on_boxed) and (t % max(1, stop_check_every) == 0):
                for i in range(B):
                    if not active[i]:
                        continue
                    w = comp_tokens[i][-stop_check_window:] if stop_check_window > 0 else comp_tokens[i]
                    s = self.decode(w)
                    if stop_on_boxed and boxed_complete(s):
                        active[i] = False
                        continue
                    if stop_on_think_close and ("</think>" in s):
                        active[i] = False
                        continue
                    if stop_on_user and (("\nUser:" in s) or ("\n\nUser:" in s)):
                        active[i] = False
                        continue

            # Use [tok, tok, ...] format to hit optimized forward_one_batch path
            step_tokens_batch = [int(x) for x in tok_cpu]
            last_logits = self.infer_model.forward_batch(step_tokens_batch, state)
            if torch.is_tensor(last_logits) and last_logits.dim() == 3:
                last_logits = last_logits[:, -1, :]

        for i in range(B):
            if bool(active[i].item()):
                truncated[i] = True

        # Clean up Rapid-Sampling tensors to free memory
        if self.cfg.use_rapid_sampling and self.sample_kernel is not None:
            del penalties, rand_states
        torch.cuda.empty_cache()

        comp_text = [self.decode(x) for x in comp_tokens]
        return comp_tokens, old_logps, comp_text, truncated


# =========================================================
# Trainer
# =========================================================

class GRPOStateTuningTrainer:
    def __init__(
        self,
        train_model,
        infer_engine: FP16BatchInference,
        encode_fn,
        decode_fn,
        data: List[Dict[str, Any]],
        out_dir: str,
        device: str,
        cfg: GRPOConfig,
        seed: int = 42,
        eval_data: List[Dict[str, Any]] = None,
    ):
        self.model = train_model
        self.infer = infer_engine
        self.encode = encode_fn
        self.decode = decode_fn
        self.data = data
        self.eval_data = eval_data if eval_data is not None else data  # Use eval_data if provided
        self.out_dir = out_dir
        self.device = device
        self.cfg = cfg
        self.rng = random.Random(seed)

        # Fixed eval indices - use a separate random generator with fixed seed for eval
        self.eval_rng = random.Random(42)  # Fixed seed for eval
        self._fixed_eval_indices = None  # Will be initialized on first eval

        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "train.log")
        self.gen_dump_path = os.path.join(out_dir, "gen_judgements.jsonl")
        self.infer_check_path = os.path.join(out_dir, "infer_check.jsonl")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")

        self._hang_f = None
        if cfg.enable_faulthandler:
            try:
                import faulthandler
                hang_path = os.path.join(out_dir, "hang_tracebacks.log")
                self._hang_f = open(hang_path, "a", encoding="utf-8", buffering=1)
                faulthandler.enable(file=self._hang_f, all_threads=True)
                if float(cfg.hang_dump_s) > 0:
                    faulthandler.dump_traceback_later(float(cfg.hang_dump_s), repeat=True, file=self._hang_f)
            except Exception:
                self._hang_f = None

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable params (expected time_state only).")

        # Use Adam instead of AdamW, eps=1e-18, no weight_decay
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr, eps=1e-18, weight_decay=0.0)

        self._ts_init: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" in n:
                    self._ts_init[n] = p.detach().clone()

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _time_state_stats(self):
        mx = 0.0
        rms_sum = 0.0
        cnt = 0
        bad = False
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "time_state" not in n:
                    continue
                if torch.isnan(p).any() or torch.isinf(p).any():
                    bad = True
                v = p.detach().float()
                mx = max(mx, float(v.abs().max().item()))
                rms_sum += float((v * v).mean().sqrt().item())
                cnt += 1
        return {"absmax": mx, "rms_avg": (rms_sum / max(1, cnt)), "bad": bad}

    def _pad_batch(self, seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, List[int]]:
        lens = [len(s) for s in seqs]
        T = max(lens)
        B = len(seqs)
        x = torch.full((B, T), pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        return x, lens

    def _compute_advantages(self, rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        return (rewards - mean) / (std + eps)

    def _ppo_clipped_objective(self, ratio: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.cfg.eps_low, 1.0 + self.cfg.eps_high) * adv
        return torch.where(adv >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))

    def _maybe_adapt_lr_by_kl(self, approx_kl: float):
        if not self.cfg.adaptive_kl:
            return
        if approx_kl is None:
            return
        lr_now = float(self.opt.param_groups[0]["lr"])
        if approx_kl > self.cfg.target_kl * 2.0:
            new_lr = max(lr_now * 0.5, 1e-6)
            if new_lr < lr_now:
                self.opt.param_groups[0]["lr"] = new_lr
                self._log(f"[KL-ADAPT] kl={approx_kl:.6f} high -> lr {lr_now:.2e} -> {new_lr:.2e}")
        elif approx_kl < self.cfg.target_kl * 0.25:
            new_lr = min(lr_now * 1.1, 5e-4)  # 上限5e-4
            if new_lr > lr_now:
                self.opt.param_groups[0]["lr"] = new_lr
                self._log(f"[KL-ADAPT] kl={approx_kl:.6f} low  -> lr {lr_now:.2e} -> {new_lr:.2e}")

    @torch.no_grad()
    def _infer_once(self, problem: str, gt: str, max_new: int, temperature: float, top_p: float, top_k: int) -> Dict[str, Any]:
        prompt = build_prompt(problem)
        ids = self.encode(prompt)
        max_prompt_len = int(self.model.args.ctx_len) - int(max_new) - 4
        max_prompt_len = max(64, max_prompt_len)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]

        comp_tokens, _, comp_texts, truncs = self.infer.generate_group_parallel(
            prompt_tokens_list=[ids],
            group_size=1,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_on_think_close=self.cfg.stop_on_think_close,
            stop_on_user=self.cfg.stop_on_user,
            stop_on_boxed=self.cfg.stop_on_boxed,
            stop_check_every=max(1, self.cfg.stop_check_every // 2),
            stop_check_window=max(64, self.cfg.stop_check_window),
        )

        txt = comp_texts[0]
        trunc = bool(truncs[0])
        if trunc:
            r = 0.0
            jdbg = {
                "pred_extracted": None,
                "gt": _strip_math_delims(gt),
                "method": "truncated_skip_judge",
                "error": None,
                "pred_parsed": None,
                "gt_parsed": None,
                "correct": False,
                "truncated_forced_zero": True,
            }
        else:
            r, jdbg0 = judge_answer(txt, gt)
            jdbg = dict(jdbg0)
            jdbg["truncated_forced_zero"] = False

        return {
            "prompt": prompt,
            "completion": txt,
            "truncated": trunc,
            "reward": float(r),
            "judge": jdbg,
            "gen_len": len(comp_tokens[0]),
        }

    @torch.no_grad()
    def sanity_infer_check(self, step: int, n: int = 3):
        for _ in range(n):
            ex = self.data[self.rng.randrange(len(self.data))]
            rec = self._infer_once(
                problem=ex.get("problem", ""),
                gt=str(ex.get("solution", "")),
                max_new=self.cfg.eval_max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            append_jsonl(self.infer_check_path, {
                "time": now_str(),
                "step": step,
                "problem": ex.get("problem", ""),
                "gt": str(ex.get("solution", "")),
                **rec
            })

    @torch.no_grad()
    def evaluate(self, step: int):
        """Batch evaluate eval_n samples in parallel with fixed questions"""
        # Initialize fixed eval indices on first call (ensures same questions every eval)
        if self._fixed_eval_indices is None:
            self._fixed_eval_indices = [self.eval_rng.randrange(len(self.eval_data)) for _ in range(self.cfg.eval_n)]
            self._log(f"[EVAL] Initialized fixed eval indices: {len(self._fixed_eval_indices)} questions from eval dataset")

        idxs = self._fixed_eval_indices

        # Build prompts for all samples using eval_data
        ex_list = [self.eval_data[i] for i in idxs]
        prompt_strs = [build_prompt(ex.get("problem", "")) for ex in ex_list]
        gts = [str(ex.get("solution", "")) for ex in ex_list]

        # Encode prompts
        max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.eval_max_new_tokens) - 4
        max_prompt_len = max(64, max_prompt_len)

        prompt_tokens_list = []
        for ps in prompt_strs:
            ids = self.encode(ps)
            if len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            prompt_tokens_list.append(ids)

        # Batch inference (group_size=1 for eval)
        comp_tokens, _, comp_texts, truncated = self.infer.generate_group_parallel(
            prompt_tokens_list=prompt_tokens_list,
            group_size=1,
            max_new_tokens=self.cfg.eval_max_new_tokens,
            temperature=self.cfg.eval_temperature,
            top_p=self.cfg.eval_top_p,
            top_k=self.cfg.eval_top_k,
            stop_on_think_close=self.cfg.stop_on_think_close,
            stop_on_user=self.cfg.stop_on_user,
            stop_on_boxed=self.cfg.stop_on_boxed,
            stop_check_every=max(1, self.cfg.stop_check_every // 2),
            stop_check_window=max(64, self.cfg.stop_check_window),
            presence_penalty=self.cfg.eval_presence_penalty,
            frequency_penalty=self.cfg.eval_frequency_penalty,
            penalty_decay=self.cfg.eval_penalty_decay,
        )

        # Parallel judge
        judge_tasks = []
        for i, (ctext, trunc, gt) in enumerate(zip(comp_texts, truncated, gts)):
            if not trunc:
                judge_tasks.append((i, ctext, gt))

        judge_results = {}
        if judge_tasks:
            with ThreadPoolExecutor(max_workers=min(32, len(judge_tasks))) as executor:
                future_to_idx = {
                    executor.submit(judge_answer, ctext, gt): i
                    for i, ctext, gt in judge_tasks
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        r, jdbg = future.result()
                        judge_results[idx] = (float(r), jdbg)
                    except Exception as e:
                        judge_results[idx] = (0.0, {"error": str(e), "correct": False})

        # Process results
        correct = 0
        trunc_cnt = 0
        lens = []
        details = []

        for i, (ex, ctext, trunc) in enumerate(zip(ex_list, comp_texts, truncated)):
            gt = gts[i]
            if trunc:
                r = 0.0
                jdbg = {"method": "truncated", "correct": False}
                trunc_cnt += 1
            else:
                r, jdbg = judge_results.get(i, (0.0, {"error": "missing", "correct": False}))

            if r >= 0.5:
                correct += 1
            lens.append(len(comp_tokens[i]))
            details.append({
                "problem": ex.get("problem", ""),
                "gt": gt,
                "completion": ctext,
                "truncated": trunc,
                "reward": float(r),
                "judge": jdbg,
                "gen_len": len(comp_tokens[i]),
            })

        acc = correct / max(1, self.cfg.eval_n)
        trunc_rate = trunc_cnt / max(1, self.cfg.eval_n)
        avg_len = sum(lens) / max(1, len(lens))

        append_jsonl(self.eval_path, {
            "time": now_str(),
            "step": step,
            "eval_n": self.cfg.eval_n,
            "acc": acc,
            "trunc_rate": trunc_rate,
            "avg_len": avg_len,
            "eval_temperature": self.cfg.eval_temperature,
            "eval_top_p": self.cfg.eval_top_p,
            "eval_top_k": self.cfg.eval_top_k,
            "eval_max_new_tokens": self.cfg.eval_max_new_tokens,
            "details": details,
        })

        self._log(f"[EVAL step {step}] acc={acc:.3f} trunc={trunc_rate:.3f} avg_len={avg_len:.1f} "
                  f"(temp={self.cfg.eval_temperature}, top_p={self.cfg.eval_top_p}, max_new={self.cfg.eval_max_new_tokens})")

    def train(self, total_steps: int):
        self._log(f"GRPO train begin: steps={total_steps} batch_prompts={self.cfg.batch_prompts} group={self.cfg.group_size} "
                  f"lr={self.cfg.lr} kl_coef={self.cfg.kl_coef} l2={self.cfg.time_state_l2} clamp={self.cfg.time_state_clamp}")
        self._log(f"Optimizer: Adam (no weight_decay), eps=1e-18, lr={self.cfg.lr}")
        self._log(f"NOTE: Original GRPO - all-0/all-1 sample filtering")
        st0 = self._time_state_stats()
        self._log(f"time_state init: absmax={st0['absmax']:.6f} rms={st0['rms_avg']:.6f} bad={st0['bad']}")

        # Eval before training (zero state baseline)
        self._log(f"[EVAL step 0] Running baseline eval with zero state...")
        self.model.eval()
        self.evaluate(step=0)

        for step in range(1, total_steps + 1):
            t0 = time.time()

            sample_cnt = 0
            sample_correct = 0
            sample_trunc = 0
            all0_cnt = 0
            all1_cnt = 0
            valid_group_cnt = 0  # 有效组计数

            # ---------------- collect batch: 持续rollout直到有效组>=min_valid_groups ----------------
            buffer = []
            rollout_round = 0

            while valid_group_cnt < self.cfg.min_valid_groups and rollout_round < self.cfg.max_rollout_rounds:
                rollout_round += 1
                ex_list = [self.data[self.rng.randrange(len(self.data))] for _ in range(self.cfg.batch_prompts)]
                prompt_strs = [build_prompt(ex.get("problem", "")) for ex in ex_list]
                gts = [str(ex.get("solution", "")) for ex in ex_list]
                probs = [ex.get("problem", "") for ex in ex_list]

                prompt_tokens_list = []
                for ps in prompt_strs:
                    ids = self.encode(ps)
                    max_prompt_len = int(self.model.args.ctx_len) - int(self.cfg.max_new_tokens) - 4
                    max_prompt_len = max(64, max_prompt_len)
                    if len(ids) > max_prompt_len:
                        ids = ids[-max_prompt_len:]
                    prompt_tokens_list.append(ids)

                comp_tokens_flat, old_logps_flat, comp_text_flat, truncated_flat = self.infer.generate_group_parallel(
                    prompt_tokens_list=prompt_tokens_list,
                    group_size=self.cfg.group_size,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                    stop_on_think_close=self.cfg.stop_on_think_close,
                    stop_on_user=self.cfg.stop_on_user,
                    stop_on_boxed=self.cfg.stop_on_boxed,
                    stop_check_every=self.cfg.stop_check_every,
                    stop_check_window=self.cfg.stop_check_window,
                )

                # Parallel judge: collect all samples needing judgment first
                judge_tasks = []  # (pi, gi, ctext, gt)
                sample_info = []  # Store all sample info for later processing

                for pi in range(self.cfg.batch_prompts):
                    pi_samples = []
                    for gi in range(self.cfg.group_size):
                        idx = pi * self.cfg.group_size + gi
                        ctoks = comp_tokens_flat[idx]
                        ologp = old_logps_flat[idx]
                        ctext = comp_text_flat[idx]
                        trunc = bool(truncated_flat[idx])
                        pi_samples.append((ctoks, ologp, ctext, trunc))
                        if not trunc:
                            judge_tasks.append((pi, gi, ctext, gts[pi]))
                    sample_info.append(pi_samples)

                # Execute judge calls in parallel
                judge_results = {}  # (pi, gi) -> (reward, jdbg)
                if judge_tasks:
                    with ThreadPoolExecutor(max_workers=min(32, len(judge_tasks))) as executor:
                        future_to_key = {
                            executor.submit(judge_answer, ctext, gt): (pi, gi)
                            for pi, gi, ctext, gt in judge_tasks
                        }
                        for future in as_completed(future_to_key):
                            pi, gi = future_to_key[future]
                            try:
                                r, jdbg0 = future.result()
                                jdbg = dict(jdbg0)
                                jdbg["truncated_forced_zero"] = False
                                judge_results[(pi, gi)] = (float(r), jdbg)
                            except Exception as e:
                                judge_results[(pi, gi)] = (0.0, {
                                    "error": str(e),
                                    "method": "parallel_judge_error",
                                    "correct": False,
                                    "truncated_forced_zero": False,
                                })

                # Process results - GRPO: keep ALL samples (no all-0/all-1 filtering)
                for pi in range(self.cfg.batch_prompts):
                    group = []
                    rewards = []
                    judges = []

                    for gi in range(self.cfg.group_size):
                        ctoks, ologp, ctext, trunc = sample_info[pi][gi]

                        if trunc:
                            r = 0.0
                            jdbg = {
                                "pred_extracted": None,
                                "gt": _strip_math_delims(gts[pi]),
                                "method": "truncated_skip_judge",
                                "error": None,
                                "pred_parsed": None,
                                "gt_parsed": None,
                                "correct": False,
                                "truncated_forced_zero": True,
                            }
                        else:
                            r, jdbg = judge_results.get((pi, gi), (0.0, {"error": "missing result"}))

                        group.append((ctoks, ologp, ctext, trunc))
                        rewards.append(float(r))
                        judges.append(jdbg)

                        sample_cnt += 1
                        if r >= 0.5:
                            sample_correct += 1
                        if trunc:
                            sample_trunc += 1

                    # Track all-0 and all-1 for logging (but don't filter them out)
                    rsum = sum(rewards)
                    is_all0 = (rsum == 0.0)
                    is_all1 = (rsum == float(self.cfg.group_size))
                    is_valid = not (is_all0 or is_all1)  # 有效组：非全0也非全1

                    if is_all0:
                        all0_cnt += 1
                    elif is_all1:
                        all1_cnt += 1
                    if is_valid:
                        valid_group_cnt += 1

                    append_jsonl(self.gen_dump_path, {
                        "time": now_str(),
                        "step": step,
                        "kept": True,  # GRPO: always keep
                        "all0": is_all0,
                        "all1": is_all1,
                        "problem": probs[pi],
                        "solution": gts[pi],
                        "prompt": prompt_strs[pi],
                        "group_size": self.cfg.group_size,
                        "max_new_tokens": self.cfg.max_new_tokens,
                        "samples": [
                            {"i": gi, "text": group[gi][2], "truncated": bool(group[gi][3]),
                             "reward": float(rewards[gi]), "judge": judges[gi]}
                            for gi in range(self.cfg.group_size)
                        ]
                    })

                    # 只保留有效组（非all0/all1）到buffer，节省内存
                    if is_valid and len(buffer) < self.cfg.min_valid_groups:
                        buffer.append((prompt_tokens_list[pi], gts[pi], group, rewards, is_valid))

                # 每轮rollout后清理显存
                torch.cuda.empty_cache()

            # 日志记录rollout情况
            if rollout_round > 1:
                self._log(f"[ROLLOUT] rounds={rollout_round}, valid_groups={valid_group_cnt}, total_sampled={rollout_round * self.cfg.batch_prompts}")

            if not buffer:
                self._log(f"WARN: empty batch (should not happen in GRPO)")
                continue

            # ---------------- build trajectories ----------------
            trajs = []
            for (prompt_tokens, gt, group, rewards_list, is_valid) in buffer:
                rewards_t = torch.tensor(rewards_list, device=self.device, dtype=torch.float32)
                adv = self._compute_advantages(rewards_t)

                for i, (comp_tokens, old_logps, _, trunc) in enumerate(group):
                    if not comp_tokens:
                        continue
                    full = prompt_tokens + comp_tokens
                    trajs.append({
                        "full_tokens": full,
                        "prompt_len": len(prompt_tokens),
                        "comp_len": len(comp_tokens),
                        "old_logps": old_logps,
                        "adv": float(adv[i].item()),
                        "reward": float(rewards_list[i]),
                        "is_valid": is_valid,  # 标记是否来自有效组
                    })

            if not trajs:
                self._log("WARN: no trajs after expansion.")
                continue

            # 分母只计算有效组(非all0/all1)的tokens
            total_comp_tokens = sum(int(tr["comp_len"]) for tr in trajs if tr["is_valid"])
            if total_comp_tokens <= 0:
                # 如果没有有效组，退回到计算所有tokens（避免除0）
                total_comp_tokens = sum(int(tr["comp_len"]) for tr in trajs)
                self._log(f"WARN: no valid groups, using all tokens for denominator")
            if total_comp_tokens <= 0:
                self._log("WARN: total_comp_tokens=0.")
                continue

            # ---------------- PPO/GRPO update ----------------
            last_loss = None
            last_kl = None
            last_clipfrac = None
            last_grad = None

            trajs_sorted = sorted(trajs, key=lambda x: len(x["full_tokens"]), reverse=True)

            # 每步开始时重置lr为初始值
            self.opt.param_groups[0]["lr"] = self.cfg.lr

            for _ep in range(self.cfg.ppo_epochs):
                self.model.train()
                self.opt.zero_grad(set_to_none=True)

                approx_kl_sum = 0.0
                approx_kl_cnt = 0
                clip_hits = 0.0
                clip_cnt = 0

                mb = max(1, int(self.cfg.micro_batch))
                for s in range(0, len(trajs_sorted), mb):
                    batch = trajs_sorted[s:s + mb]
                    seqs = [b["full_tokens"] for b in batch]
                    padded, _ = self._pad_batch(seqs, pad_id=0)

                    inp = padded[:, :-1].contiguous()
                    tgt = padded[:, 1:].contiguous()

                    logits = self.model(inp)
                    if torch.is_tensor(logits) and logits.dim() == 2:
                        logits = logits.unsqueeze(0)

                    logp = F.log_softmax(logits.float(), dim=-1)
                    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).contiguous()

                    obj_sum = 0.0
                    kl_sum_tok = 0.0
                    ts_reg = 0.0

                    for bi, tr in enumerate(batch):
                        prompt_len = int(tr["prompt_len"])
                        comp_len = int(tr["comp_len"])
                        old_lp = torch.tensor(tr["old_logps"], device=self.device, dtype=torch.float32)
                        adv = torch.tensor(tr["adv"], device=self.device, dtype=torch.float32)

                        start = prompt_len - 1
                        end = start + comp_len

                        new_lp = picked[bi, start:end].to(torch.float32)

                        if new_lp.numel() != old_lp.numel():
                            m = min(new_lp.numel(), old_lp.numel())
                            new_lp = new_lp[:m]
                            old_lp = old_lp[:m]
                            if m <= 0:
                                continue

                        log_ratio = new_lp - old_lp
                        ratio = torch.exp(log_ratio)

                        obj = self._ppo_clipped_objective(ratio, adv)
                        obj_sum = obj_sum + obj.sum()

                        kl_tok = 0.5 * (log_ratio ** 2)
                        kl_sum_tok = kl_sum_tok + kl_tok.sum()

                        with torch.no_grad():
                            approx_kl_sum += float(kl_tok.mean().item())
                            approx_kl_cnt += 1
                            clipped = (ratio < (1.0 - self.cfg.eps_low)) | (ratio > (1.0 + self.cfg.eps_high))
                            clip_hits += float(clipped.float().mean().item())
                            clip_cnt += 1

                    if self.cfg.time_state_l2 > 0:
                        for n, p in self.model.named_parameters():
                            if "time_state" in n:
                                ts_reg = ts_reg + (p.float() - self._ts_init[n].float()).pow(2).mean()

                    loss = -(obj_sum / float(total_comp_tokens))
                    loss = loss + float(self.cfg.kl_coef) * (kl_sum_tok / float(total_comp_tokens))
                    if self.cfg.time_state_l2 > 0:
                        loss = loss + float(self.cfg.time_state_l2) * ts_reg

                    loss.backward()
                    last_loss = float(loss.detach().item())

                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg.grad_clip
                    )

                with torch.no_grad():
                    g2 = 0.0
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            g = p.grad.detach().float()
                            g2 += float((g.norm(2) ** 2).item())
                    last_grad = math.sqrt(g2)

                self.opt.step()

                if self.cfg.time_state_clamp and self.cfg.time_state_clamp > 0:
                    cv = float(self.cfg.time_state_clamp)
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if "time_state" in n:
                                p.data.clamp_(-cv, cv)

                last_kl = float(approx_kl_sum / max(1, approx_kl_cnt))
                last_clipfrac = float(clip_hits / max(1, clip_cnt))

                # KL early stop: 如果KL过大，提前终止PPO epochs
                if last_kl > self.cfg.target_kl * self.cfg.kl_early_stop_factor:
                    self._log(f"[KL-EARLY-STOP] epoch {_ep+1}/{self.cfg.ppo_epochs}: kl={last_kl:.6f} > {self.cfg.target_kl * self.cfg.kl_early_stop_factor:.6f}, stopping PPO epochs")
                    break

            # ---------------- logging / save / eval ----------------
            avg_r = sum(tr["reward"] for tr in trajs) / max(1, len(trajs))
            dt = time.time() - t0
            st = self._time_state_stats()
            lr_now = float(self.opt.param_groups[0]["lr"])

            if step % self.cfg.log_interval == 0:
                self._log(
                    f"[step {step}/{total_steps}] "
                    f"samples={sample_cnt} (all0={all0_cnt}, all1={all1_cnt}) | "
                    f"acc={sample_correct/max(1,sample_cnt):.3f} trunc={sample_trunc/max(1,sample_cnt):.3f} | "
                    f"avg_reward={avg_r:.4f} loss={last_loss} grad={last_grad:.3f} "
                    f"approx_kl={last_kl:.6f} clip_frac={last_clipfrac:.3f} | "
                    f"ts(absmax={st['absmax']:.4f}, rms={st['rms_avg']:.4f}, bad={st['bad']}) | "
                    f"hp(lr={lr_now:.2e}, kl_coef={self.cfg.kl_coef}, l2={self.cfg.time_state_l2}, clamp={self.cfg.time_state_clamp}, "
                    f"temp={self.cfg.temperature}, top_p={self.cfg.top_p}, max_new={self.cfg.max_new_tokens}) "
                    f"step_time={dt:.1f}s"
                )

            if step % self.cfg.infer_check_interval == 0:
                self.model.eval()
                self.sanity_infer_check(step=step, n=3)

            if step % self.cfg.eval_interval == 0:
                self.model.eval()
                self.evaluate(step=step)

            if step % self.cfg.save_interval == 0 or step == total_steps:
                ckpt_path = os.path.join(self.out_dir, f"ckpt_step{step}.pth")
                torch.save({
                    "time": now_str(),
                    "step": step,
                    "cfg": self.cfg.__dict__,
                    "time_state": {n: p.detach().cpu() for n, p in self.model.named_parameters() if "time_state" in n},
                }, ckpt_path)

                latest_ts_path = os.path.join(self.out_dir, "latest_time_state.pth")
                save_time_state_only(self.model, latest_ts_path)

                self._log(f"saved: {ckpt_path}")
                self._log(f"saved: {latest_ts_path}")

        if self.cfg.enable_faulthandler and float(self.cfg.hang_dump_s) > 0:
            try:
                import faulthandler
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass

        self._log("train end.")


# =========================================================
# Sanity check: train vs infer last-logits top1
# =========================================================

@torch.no_grad()
def sanity_check_train_vs_fp16(train_model, infer_model, infer_engine, encode, decode, train_device="cuda:0", infer_device="cuda:1"):
    prompt = "User: What is 2+2? think\nAssistant: <think>\n"
    ids = encode(prompt)
    if not ids:
        raise RuntimeError("encode(prompt) returned empty")

    # Train model forward on train_device
    t = torch.tensor([ids], device=train_device, dtype=torch.long)
    logits_train = train_model(t)[0, -1].float().cpu()

    # Infer model forward on infer_device
    state = infer_engine.init_state_with_time_state(B=1)
    out2 = infer_model.forward_batch([ids], state)
    if torch.is_tensor(out2) and out2.dim() == 3:
        out2 = out2[:, -1, :]
    logits_infer = out2[0].float().cpu()

    top_train = int(torch.argmax(logits_train).item())
    top_infer = int(torch.argmax(logits_infer).item())

    print("[sanity] top1_train =", top_train, "->", repr(decode([top_train])), flush=True)
    print("[sanity] top1_infer =", top_infer, "->", repr(decode([top_infer])), flush=True)

    if top_train != top_infer:
        # For dual-GPU mode, allow mismatch as warning due to fp16/bf16 numerical differences
        print("[sanity] WARNING: train vs infer top1 mismatch - may be due to fp16/bf16 precision diff", flush=True)
        # Check if top5 overlaps
        top5_train = torch.topk(logits_train, 5).indices.tolist()
        top5_infer = torch.topk(logits_infer, 5).indices.tolist()
        overlap = len(set(top5_train) & set(top5_infer))
        print(f"[sanity] top5_train = {top5_train}", flush=True)
        print(f"[sanity] top5_infer = {top5_infer}", flush=True)
        print(f"[sanity] top5 overlap = {overlap}/5", flush=True)
        if overlap < 3:
            raise RuntimeError("SANITY FAIL: train vs infer top5 has <3 overlap - check model loading")


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/state-tuning/rwkv7-g1c-2.9b-20251231-ctx8192", help="model path")
    ap.add_argument("--train_jsonl", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/state-tuning/gsmk8ktrain.parquet")
    ap.add_argument("--eval_data", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/state-tuning/gsmk_test.json", help="eval dataset path")
    ap.add_argument("--out_dir", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/state-tuning/out_grpo_2.9b_run2")

    ap.add_argument("--total_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--grad_cp", type=int, default=1, help="Gradient checkpointing: 0=off, 1=on (saves memory)")

    # GRPO: batch_prompts=8, group_size=16 -> 128 outputs per step
    ap.add_argument("--batch_prompts", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--collect_rounds", type=int, default=1, help="Number of sampling rounds per step")
    ap.add_argument("--min_valid_groups", type=int, default=4, help="最少有效组数，达到后停止rollout")
    ap.add_argument("--max_rollout_rounds", type=int, default=10, help="最大rollout轮数")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--decay", type=float, default=0.995)
    ap.add_argument("--mask_token0", action="store_true")

    # Rapid-Sampling repetition penalty
    ap.add_argument("--use_rapid_sampling", action="store_true", default=True)
    ap.add_argument("--presence_penalty", type=float, default=0.8)
    ap.add_argument("--repetition_penalty", type=float, default=0.6)
    ap.add_argument("--penalty_decay", type=float, default=0.995)

    ap.add_argument("--ppo_epochs", type=int, default=2)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eps_low", type=float, default=0.2)
    ap.add_argument("--eps_high", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=0.2)

    ap.add_argument("--kl_coef", type=float, default=0.001)
    ap.add_argument("--target_kl", type=float, default=0.01)
    ap.add_argument("--adaptive_kl", action="store_true", default=True)
    ap.add_argument("--kl_early_stop_factor", type=float, default=1.5, help="PPO epoch提前停止因子")

    ap.add_argument("--time_state_l2", type=float, default=1e-6)
    ap.add_argument("--time_state_clamp", type=float, default=0.0)

    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument("--save_interval", type=int, default=20)
    ap.add_argument("--infer_check_interval", type=int, default=50)

    ap.add_argument("--eval_interval", type=int, default=20)
    ap.add_argument("--eval_n", type=int, default=128)
    ap.add_argument("--eval_temperature", type=float, default=0.3)
    ap.add_argument("--eval_top_p", type=float, default=0.4)
    ap.add_argument("--eval_top_k", type=int, default=500)
    ap.add_argument("--eval_max_new_tokens", type=int, default=2048)
    ap.add_argument("--eval_presence_penalty", type=float, default=0.5)
    ap.add_argument("--eval_frequency_penalty", type=float, default=0.1)
    ap.add_argument("--eval_penalty_decay", type=float, default=0.99)

    ap.add_argument("--state_init", type=str, default=None, help="optional: load time_state-only checkpoint")
    ap.add_argument("--tokenizer", type=str, default="/data_temp/mnt/raid5/zjx/rwkv/inference/sota/reference/rwkv_vocab_v20230424.txt")

    ap.add_argument("--enable_faulthandler", action="store_true")
    ap.add_argument("--hang_dump_s", type=float, default=0.0)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Use cuda:0 when CUDA_VISIBLE_DEVICES is set, otherwise cuda:1
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0"  # When CUDA_VISIBLE_DEVICES is set, device 0 is the selected GPU
        infer_device = "cuda:0"
    else:
        device = "cuda:1"  # Prefer cuda:1 without CUDA_VISIBLE_DEVICES
        infer_device = "cuda:1"

    os.environ["RWKV_HEAD_SIZE_A"] = str(HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "state"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_data(args.train_jsonl)
    if not data:
        raise RuntimeError("empty train data")

    # Load eval data
    eval_data = None
    if args.eval_data:
        eval_data = load_data(args.eval_data)
        print(f"Loaded eval data: {len(eval_data)} samples from {args.eval_data}", flush=True)
    else:
        print("No eval_data specified, using train data for eval", flush=True)

    from utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(args.tokenizer)

    encode = lambda s: tok.encode(s)

    def safe_decode(ids):
        try:
            return tok.decode(ids, utf8_errors="replace")
        except TypeError:
            pass
        try:
            return tok.decode(ids)
        except UnicodeDecodeError:
            try:
                b = tok.decodeBytes(ids)
                return b.decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(int(x) % 256) for x in ids)

    decode = safe_decode

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    print(f"Loading model: {pth_path}", flush=True)
    print(f"Train device: {device}, Infer device: {infer_device}", flush=True)
    train_model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=int(args.ctx_len), grad_cp=int(args.grad_cp))
    infer_model, _ = load_infer_model_fp16(base_name, device=infer_device)

    trainable = freeze_except_time_state(train_model)
    if trainable <= 0:
        raise RuntimeError("No trainable time_state found. Check model weights / naming.")
    print(f"Trainable parameters (time_state): {trainable}", flush=True)

    if args.state_init:
        ok = load_time_state_only(train_model, args.state_init)
        print(f"[state_init] loaded={ok} from {args.state_init}", flush=True)

    cfg = GRPOConfig(
        batch_prompts=int(args.batch_prompts),
        group_size=int(args.group_size),
        collect_rounds=int(args.collect_rounds),
        min_valid_groups=int(args.min_valid_groups),
        max_rollout_rounds=int(args.max_rollout_rounds),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        decay=float(args.decay),
        mask_token0=bool(args.mask_token0),

        # Rapid-Sampling
        use_rapid_sampling=bool(args.use_rapid_sampling),
        presence_penalty=float(args.presence_penalty),
        repetition_penalty=float(args.repetition_penalty),
        penalty_decay=float(args.penalty_decay),

        ppo_epochs=int(args.ppo_epochs),
        micro_batch=int(args.micro_batch),
        lr=float(args.lr),
        eps_low=float(args.eps_low),
        eps_high=float(args.eps_high),
        grad_clip=float(args.grad_clip),

        kl_coef=float(args.kl_coef),
        target_kl=float(args.target_kl),
        adaptive_kl=bool(args.adaptive_kl),
        kl_early_stop_factor=float(args.kl_early_stop_factor),

        time_state_l2=float(args.time_state_l2),
        time_state_clamp=float(args.time_state_clamp),

        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        infer_check_interval=int(args.infer_check_interval),

        eval_interval=int(args.eval_interval),
        eval_n=int(args.eval_n),
        eval_temperature=float(args.eval_temperature),
        eval_top_p=float(args.eval_top_p),
        eval_top_k=int(args.eval_top_k),
        eval_max_new_tokens=int(args.eval_max_new_tokens),
        eval_presence_penalty=float(args.eval_presence_penalty),
        eval_frequency_penalty=float(args.eval_frequency_penalty),
        eval_penalty_decay=float(args.eval_penalty_decay),

        enable_faulthandler=bool(args.enable_faulthandler),
        hang_dump_s=float(args.hang_dump_s),
    )

    infer_engine = FP16BatchInference(infer_model, train_model, encode, decode, device=infer_device, cfg=cfg)

    sanity_check_train_vs_fp16(train_model, infer_model, infer_engine, encode, decode, train_device=device, infer_device=infer_device)

    trainer = GRPOStateTuningTrainer(
        train_model=train_model,
        infer_engine=infer_engine,
        encode_fn=encode,
        decode_fn=decode,
        data=data,
        out_dir=args.out_dir,
        device=device,
        cfg=cfg,
        seed=int(args.seed),
        eval_data=eval_data,
    )

    trainer.train(total_steps=int(args.total_steps))


if __name__ == "__main__":
    main()
