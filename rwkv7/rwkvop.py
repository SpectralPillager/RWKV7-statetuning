"""
RWKV Operator - 支持State-Tuning的版本

修改说明：
1. 添加了wkv7_state CUDA kernel加载（用于state-tuning）
2. RUN_RWKV7_STATE使用CUDA实现，支持梯度回传到初始state
"""

from einops import rearrange
import os, math, gc, importlib
import torch
import torch.nn.functional as F

########################################################################################################
# CUDA Kernel
########################################################################################################

HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", "64"))
CHUNK_LEN = 16


def RUN_CUDA_RWKV7g():
    raise NotImplementedError('RUN_CUDA_RWKV7g not implemented')

def RUN_RWKV7_STATE():
    raise NotImplementedError('RUN_RWKV7_STATE not implemented')

# ========================================================================
# CUDA Backend
# ========================================================================
from torch.utils.cpp_extension import load


if 'x070' in os.environ.get("RWKV_MY_TESTING", ""):

    # 加载标准WKV7 kernel
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=['cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b, HEAD_SIZE=64):
        B,T,HC = q.shape
        C = HEAD_SIZE
        H = HC // C

        # Padding
        orig_T = T
        if T % CHUNK_LEN != 0:
            pad_len = CHUNK_LEN - (T % CHUNK_LEN)
            q = F.pad(q, (0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            a = F.pad(a, (0, 0, 0, pad_len))
            b = F.pad(b, (0, 0, 0, pad_len))
            T = T + pad_len

        q,w,k,v,a,b = [i.view(B,T,H,C) for i in [q,w,k,v,a,b]]
        y = WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

        if T != orig_T:
            y = y[:, :orig_T, :].contiguous()
        return y

    # ====================================================================
    # WKV7 State - 支持初始state的CUDA实现
    # ====================================================================

    # 加载state版本kernel
    if os.environ.get("RWKV_TRAIN_TYPE") in ['state', 'fullstate']:
        print("Loading WKV7 State CUDA kernel for state-tuning...")
        load(name="wkv7_state", sources=['cuda/wkv7state_cuda.cu', 'cuda/wkv7state_op.cpp'], 
             is_python_module=False, verbose=True, extra_cuda_cflags=flags)

        class WKV7StateFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, w, q, k, v, z, b, s0):
                """
                s0: (H, C, C) - 原始state参数（不是扩展后的）
                """
                B, T, H, C = w.shape
                assert T % CHUNK_LEN == 0
                assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b, s0])
                assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
                assert s0.shape == (H, C, C), f"s0 shape should be (H, C, C)={(H, C, C)}, got {s0.shape}"

                # 在function内部扩展state
                s0_expanded = s0.unsqueeze(0).expand(B, H, C, C).contiguous()

                y = torch.empty_like(v)
                s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
                sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
                sT = torch.empty(B, H, C, C, dtype=w.dtype, device=w.device)  # final state

                torch.ops.wkv7_state.forward(w, q, k, v, z, b, s0_expanded, y, s, sa, sT)
                ctx.save_for_backward(w, q, k, v, z, b, s, sa, s0_expanded)
                return y

            @staticmethod
            def backward(ctx, dy):
                assert dy.dtype == torch.bfloat16
                assert dy.is_contiguous()
                w, q, k, v, z, b, s, sa, s0_expanded = ctx.saved_tensors
                B, T, H, C = w.shape

                dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
                ds0 = torch.empty(B, H, C, C, dtype=torch.bfloat16, device=w.device)

                torch.ops.wkv7_state.backward(w, q, k, v, z, b, dy, s, sa, s0_expanded, dw, dq, dk, dv, dz, db, ds0)

                # 对batch维度求和，返回(H, C, C)形状 - 与输入s0形状一致
                ds0_sum = ds0.sum(dim=0)

                return dw, dq, dk, dv, dz, db, ds0_sum

        def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64):
            """
            带初始state的RWKV7 CUDA实现

            Args:
                r, k, v, w, a, b: (B, T, HC) 输入张量
                s: (H, C, C) 初始state（可训练参数）
                HEAD_SIZE: head维度，默认64

            Returns:
                output: (B, T, HC)
                state: None（当前不返回最终state）
            """
            B, T, HC = r.shape
            C = HEAD_SIZE
            H = HC // C

            # Padding T to multiple of CHUNK_LEN
            orig_T = T
            if T % CHUNK_LEN != 0:
                pad_len = CHUNK_LEN - (T % CHUNK_LEN)
                r = F.pad(r, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                w = F.pad(w, (0, 0, 0, pad_len))
                a = F.pad(a, (0, 0, 0, pad_len))
                b = F.pad(b, (0, 0, 0, pad_len))
                T = T + pad_len

            # Reshape to (B, T, H, C)
            r_4d = r.view(B, T, H, C).contiguous()
            k_4d = k.view(B, T, H, C).contiguous()
            v_4d = v.view(B, T, H, C).contiguous()
            w_4d = w.view(B, T, H, C).contiguous()
            a_4d = a.view(B, T, H, C).contiguous()
            b_4d = b.view(B, T, H, C).contiguous()

            # state保持(H, C, C)形状，WKV7StateFunction内部会扩展
            s_3d = s.contiguous()

            # 确保dtype为bfloat16
            if r_4d.dtype != torch.bfloat16:
                r_4d = r_4d.to(torch.bfloat16)
                k_4d = k_4d.to(torch.bfloat16)
                v_4d = v_4d.to(torch.bfloat16)
                w_4d = w_4d.to(torch.bfloat16)
                a_4d = a_4d.to(torch.bfloat16)
                b_4d = b_4d.to(torch.bfloat16)
            if s_3d.dtype != torch.bfloat16:
                s_3d = s_3d.to(torch.bfloat16)

            # 调用CUDA kernel
            # 参数顺序: (w, q, k, v, z, b, s0) 其中 q=r, z=a
            y = WKV7StateFunction.apply(w_4d, r_4d, k_4d, v_4d, a_4d, b_4d, s_3d)

            # Reshape back
            y = y.view(B, T, HC)
            if T != orig_T:
                y = y[:, :orig_T, :].contiguous()

            return y, None

elif 'x060' in os.environ.get("RWKV_MY_TESTING", ""):
    # RWKV6 实现
    pass
else:
    # RWKV5 实现
    pass

