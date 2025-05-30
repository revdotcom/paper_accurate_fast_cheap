########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
#import pytorch_lightning as pl
#from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
#from pytorch_lightning.strategies import DeepSpeedStrategy
#if importlib.util.find_spec('deepspeed'):
#    import deepspeed
#    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
import os
module_dir = os.path.dirname(os.path.relpath(__file__)) + "/.."

# from rwkvkit.ops.rwkv6_jp import native_recurrent_rwkv6

def __nop(ob):
    return ob

# def cuda_warpper(r, k, v, w, u):
    # return WKV_6.apply(r, k, v, w, u)


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

MySilentSwitch = __nop
if os.environ.get('TORCHSCRIPT_EXPORT', '0') == '1':
    MySilentSwitch = torch.jit.unused

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'x060' in os.environ["RWKV_MY_TESTING"]:
    if os.environ["RWKV_TRAIN_TYPE"] == 'states':
        wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
            
        class WKV_6STATE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    assert s.dtype == torch.bfloat16
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    assert s.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u, s)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u, s = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    gs = torch.sum(gs, 0).view(H, C//H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu, gs)

        @MySilentSwitch
        def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
            return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
    else:
        load(name="wkv6", sources=[f"{module_dir}/cuda/wkv6_op.cpp", f"{module_dir}/cuda/wkv6_cuda.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
            
        class WKV_6(torch.autograd.Function):

            @staticmethod
            def forward(ctx, r, k, v, w, u):
                with torch.no_grad():
                    B, T, C = r.size()
                    H = C // HEAD_SIZE
                    assert C % HEAD_SIZE == 0
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (gr, gk, gv, gw, gu)

        # @torch.jit.script_method
        #@torch.jit.ignore
        # @torch.jit.unused
        @MySilentSwitch
        def RUN_CUDA_RWKV6(r, k, v, w, u):
            return WKV_6.apply(r, k, v, w, u)

        class WKV_6_FP32(torch.autograd.Function):

            @staticmethod
            def forward(ctx, r, k, v, w, u):
                with torch.no_grad():   
                    #print("WKV_6_FP32 forward")
                    B, T, C = r.size()
                    H = C // HEAD_SIZE
                    assert C % HEAD_SIZE == 0
                    assert r.dtype == torch.float32
                    assert k.dtype == torch.float32
                    assert v.dtype == torch.float32
                    assert w.dtype == torch.float32
                    assert u.dtype == torch.float32
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv6.forward_fp32(B, T, C, H, r, k, v, w, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.float32
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv6.backward_fp32(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (gr, gk, gv, gw, gu)

        # @torch.jit.script_method
        #@torch.jit.ignore
        # @torch.jit.unused
        @MySilentSwitch
        def RUN_CUDA_RWKV6_FP32(r, k, v, w, u):
            #print("RUN_CUDA_RWKV6_FP32")
            return WKV_6_FP32.apply(r, k, v, w, u)

    
########################################################################################################
class RWKV_Tmix_x060c(MyModule):
    def __init__(self, head_size : int, n_layers : int, n_embd : int, dim_att : int, layer_id : int):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = dim_att // self.head_size
        self.dim_att = dim_att
        self.n_embd = n_embd
        self.n_layers = n_layers
        assert self.dim_att % self.n_head == 0

        self.torchscript_friendly = os.environ.get('TORCHSCRIPT_EXPORT', '0') == '1'

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (self.n_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            D_MIX_LORA = 32
            self.time_maa_rkvw_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MIX_LORA*4))
            self.time_maa_rkvw_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, self.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(self.n_embd, self.dim_att, bias=False)

        self.value = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(self.dim_att)

    @MyFunction
    def forward(self, x : torch.Tensor) -> torch.Tensor :
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_rkvw_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkvw_w2).view(4, B, T, C)

        r, k, v, w = xxx.unbind(dim=0)
        r = x + xx * (self.time_maa_r + r)
        k = x + xx * (self.time_maa_k + k)
        v = x + xx * (self.time_maa_v + v)
        w = x + xx * (self.time_maa_w + w)
        
        r = self.receptance(r)
        k = self.key(k)
        v = self.value(v)
        w = self.time_decay + torch.tanh(w @ self.time_decay_w1) @ self.time_decay_w2

        # k = k * (1-(-w.exp()).exp()) # for fp32
        #k = k * (1-(-w.float().exp()).exp()).to(dtype=torch.bfloat16) # for bf16
        # when training

        #x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        if x.dtype == torch.float32:
            x = RUN_CUDA_RWKV6_FP32(r, k, v, w, u=self.time_faaaa)
        else:   
            x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        # if not self.torchscript_friendly:
        #     x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        # else:
        #     # when exporting
        #     with torch.no_grad():
        #         B, T, C = r.size()
        #         H = C // self.head_size
        #         u = self.time_faaaa
        #         assert C % self.head_size == 0
        #         assert r.dtype == torch.bfloat16
        #         assert k.dtype == torch.bfloat16
        #         assert v.dtype == torch.bfloat16
        #         assert w.dtype == torch.bfloat16
        #         assert u.dtype == torch.bfloat16
        #         assert r.is_contiguous()
        #         assert k.is_contiguous()
        #         assert v.is_contiguous()
        #         assert w.is_contiguous()
        #         assert u.is_contiguous()
        #         y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
        #         torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
        #         x = y

        x = self.ln_x(x)
        x = self.output(x)
        return x
    

## This seems to be what we want
class RWKV_Tmix_x060c_orig(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            D_MIX_LORA = 32
            self.time_maa_rkvw_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*4))
            self.time_maa_rkvw_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(args.dim_att)

    @MyFunction
    def forward(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_rkvw_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkvw_w2).view(4, B, T, C)

        r, k, v, w = xxx.unbind(dim=0)
        r = x + xx * (self.time_maa_r + r)
        k = x + xx * (self.time_maa_k + k)
        v = x + xx * (self.time_maa_v + v)
        w = x + xx * (self.time_maa_w + w)
        
        r = self.receptance(r)
        k = self.key(k)
        v = self.value(v)
        w = self.time_decay + torch.tanh(w @ self.time_decay_w1) @ self.time_decay_w2

        # k = k * (1-(-w.exp()).exp()) # for fp32
        k = k * (1-(-w.float().exp()).exp()).to(dtype=torch.bfloat16) # for bf16

        x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        # x, _ = native_recurrent_rwkv6(r, k, v, w, u=self.time_faaaa)

        # x = self.wkv.RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        # x = cuda_warpper(r, k, v, w, u=self.time_faaaa)

        x = self.ln_x(x)
        x = self.output(x)
        return x
    
########################################################################################################
class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

########################################################################################################

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'x060a' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x060a(args, layer_id)
            elif 'x060b' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x060b(args, layer_id)
            elif 'x060' in os.environ["RWKV_MY_TESTING"]:
                if os.environ["RWKV_TRAIN_TYPE"] == 'states':
                    self.att = RWKV_Tmix_x060_state(args, layer_id)
                else:
                    self.att = RWKV_Tmix_x060(args, layer_id)
            elif 'x052' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_Tmix_x052(args, layer_id)
            elif 'mamba' in os.environ["RWKV_MY_TESTING"]:
                self.att = Mamba(d_model=args.n_embd, d_state=16, d_conv=4, expand=2.125) # match rwkv6 #params

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        elif 'x060' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = RWKV_CMix_x060(args, layer_id)
        elif 'x052' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = RWKV_CMix_x052(args, layer_id)
        elif 'mamba' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = Mamba(d_model=args.n_embd, d_state=16, d_conv=4, expand=2.125) # match rwkv6 #params
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.args.dropout == 0:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

