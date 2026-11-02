"""Frozen node linears using tensor-core products and FP32 accumulation.

TF32x3 and residual FP16x3 use the FP32 checkpoint weights. FP16w2 instead
requires exactly FP16-representable weights (an FP16-quantized checkpoint)
and splits only the FP32 activation. Forward and coordinate VJP use the same
declared arithmetic; ordinary differentiable linears remain the training path.
"""
import torch
from torch import nn
from torch.nn import functional as F
import triton as tr
import triton.language as tl


@tr.jit
def _accurate_dot(a, b, acc, PRECISION: tl.constexpr):
    if PRECISION == 'tf32x3':
        return acc + tl.dot(a, b, input_precision='tf32x3')
    elif PRECISION == 'fp16w2':
        # B is the FP16 weight tensor of an FP16-quantized checkpoint.
        # Only the FP32 activation needs a high/residual split. Both products
        # accumulate in FP32 and the coordinate VJP uses the same weights.
        amax = tl.maximum(tl.max(tl.max(tl.abs(a), 0), 0), 1.e-30)
        sa = tl.exp2(10. - tl.floor(tl.log2(amax)))
        av = a*sa
        ah = av.to(tl.float16)
        al = (av-ah.to(tl.float32)).to(tl.float16)
        return acc+tl.dot(ah, b, tl.dot(al, b))/sa
    else:
        # Split each FP32 operand into a high and residual FP16 value. A
        # power-of-two tile scale keeps both large and small operands in
        # range; three products keep all first-order residual terms. The
        # dropped residual*residual term is O(2^-22), not FP16 precision.
        amax = tl.maximum(tl.max(tl.max(tl.abs(a), 0), 0), 1.e-30)
        bmax = tl.maximum(tl.max(tl.max(tl.abs(b), 0), 0), 1.e-30)
        sa = tl.exp2(10. - tl.floor(tl.log2(amax)))
        sb = tl.exp2(10. - tl.floor(tl.log2(bmax)))
        av, bv = a * sa, b * sb
        ah, bh = av.to(tl.float16), bv.to(tl.float16)
        al = (av - ah.to(tl.float32)).to(tl.float16)
        bl = (bv - bh.to(tl.float32)).to(tl.float16)
        low = tl.dot(al, bh)
        low = tl.dot(ah, bl, low)
        value = tl.dot(ah, bh, low)
        return acc + value / (sa * sb)


@tr.jit(do_not_specialize=["M"])
def _mm(A,B,C,M,N:tl.constexpr,K:tl.constexpr,AS0:tl.constexpr,AS1:tl.constexpr,
        BS0:tl.constexpr,BS1:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
        PRECISION:tl.constexpr='tf32x3'):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    column=tl.program_id(1)*BN+tl.arange(0,BN)
    base=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+base
        a=tl.load(A+row[:,None]*AS0+k[None,:]*AS1,(row[:,None]<M)&(k[None,:]<K),0)
        b=tl.load(B+k[:,None]*BS0+column[None,:]*BS1,(k[:,None]<K)&(column[None,:]<N),0)
        acc=_accurate_dot(a,b,acc,PRECISION)
    tl.store(C+row[:,None]*N+column[None,:],acc,(row[:,None]<M)&(column[None,:]<N))


def _node_matmul_tile(m, n, k, b_column_stride, precision):
    """Shape/layout rule tuned on large frozen-node products; no model lookup.

    Small matrices and the FP32/TF32 path keep the default layout. The
    reduction width changes tensor-core tiling, not the declared arithmetic.
    """
    bm, bn, bk = 64, 32 if n <= 32 else 64, 32
    if m < 8192 or precision not in ('fp16w2', 'fp16x3'):
        return bm, bn, bk
    if k == 16:
        bk = 16
    elif precision == 'fp16x3':
        if n <= 32:
            bm = 128
    elif k == 64:
        bk = 64
        if n <= 32:
            bm = 128
        elif n >= 128:
            bn = 128
    elif 64 < k <= 128:
        bk = 128
        if k % 32 == 0 and n >= 128 and b_column_stride == 1:
            bn, bk = 128, 32
    elif k > 128 and n >= 128 and b_column_stride == 1:
        bn = 128
    return bm, bn, bk


def matmul(a,b,precision='tf32x3'):
    m,k=a.shape;n=b.shape[1]
    if b.shape[0]!=k:raise ValueError("Incompatible matrix shapes")
    if not a.is_cuda or a.dtype!=torch.float32 or b.dtype!=(torch.float16 if precision=='fp16w2' else torch.float32):
        raise ValueError('Node products require CUDA FP32 activations and the declared weight dtype')
    if precision not in ('tf32x3','fp16x3','fp16w2'):
        raise ValueError('Unsupported node multiplication precision')
    result=a.new_empty(m,n)
    if m and n:
        bm,bn,bk=(_node_matmul_tile(m,n,k,b.stride(1),precision) if a.is_contiguous()
                  else (64,32 if n<=32 else 64,32))
        _mm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,b,result,m,n,k,*a.stride(),*b.stride(),
            bm,bn,bk,precision,num_warps=4,num_stages=3)
    return result


@tr.jit(do_not_specialize=['M'])
def _mm_grouped(A,B,C,M,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,
                AS0:tl.constexpr,AS1:tl.constexpr,AS2:tl.constexpr,
                BS0:tl.constexpr,BS1:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                PRECISION:tl.constexpr='tf32x3'):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    column=tl.program_id(1)*BN+tl.arange(0,BN)
    base=tl.arange(0,BK)
    offset=(row//G)*AS0+(row%G)*AS1
    acc=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+base
        a=tl.load(A+offset[:,None]+k[None,:]*AS2,(row[:,None]<M)&(k[None,:]<K),0.)
        b=tl.load(B+k[:,None]*BS0+column[None,:]*BS1,(k[:,None]<K)&(column[None,:]<N),0.)
        acc=_accurate_dot(a,b,acc,PRECISION)
    tl.store(C+row[:,None]*N+column[None,:],acc,(row[:,None]<M)&(column[None,:]<N))


def grouped_matmul(a,b,precision='tf32x3'):
    """Flatten only logical rows; avoid copies for sliced Cartesian channels."""
    if a.ndim!=3 or a.is_contiguous():return matmul(a.reshape(-1,a.shape[-1]).contiguous(),b,precision)
    if a.dtype!=torch.float32 or b.dtype!=(torch.float16 if precision=='fp16w2' else torch.float32) or not a.is_cuda:
        raise ValueError('Strided node product dtype differs from the declared precision')
    batches,groups,k=a.shape;m=batches*groups;n=b.shape[1]
    if b.shape[0]!=k:raise ValueError('Incompatible matrix shapes')
    result=a.new_empty(m,n)
    if m and n:
        bn=32 if n<=32 else 64
        _mm_grouped[(tr.cdiv(m,64),tr.cdiv(n,bn))](a,b,result,m,n,k,groups,
            *a.stride(),*b.stride(),64,bn,32,precision,num_warps=4)
    return result


@tr.jit(do_not_specialize=['M'])
def _mm_silu(A,W,Bias,Pre,Out,M,N:tl.constexpr,K:tl.constexpr,
             HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
             PRECISION:tl.constexpr='tf32x3'):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    column=tl.program_id(1)*BN+tl.arange(0,BN)
    base=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK+base
        a=tl.load(A+row[:,None]*K+k[None,:],(row[:,None]<M)&(k[None,:]<K),0.)
        w=tl.load(W+k[:,None]+column[None,:]*K,(k[:,None]<K)&(column[None,:]<N),0.)
        acc=_accurate_dot(a,w,acc,PRECISION)
    if HAS_BIAS:acc+=tl.load(Bias+column,column<N,0.)[None,:]
    mask=(row[:,None]<M)&(column[None,:]<N)
    address=row[:,None]*N+column[None,:]
    tl.store(Pre+address,acc,mask)
    tl.store(Out+address,acc/(1.+tl.exp(-acc)),mask)


@tr.jit(do_not_specialize=['N'])
def _silu_gradient(Pre,Gradient,Out,N,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(Pre+index,index<N,0.)
    g=tl.load(Gradient+index,index<N,0.)
    sigmoid=1./(1.+tl.exp(-x))
    tl.store(Out+index,g*(sigmoid+x*sigmoid*(1.-sigmoid)),index<N)


class _LinearSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,precision='tf32x3'):
        shape=x.shape;a=x.reshape(-1,shape[-1]).contiguous();w=weight.contiguous()
        m,k=a.shape;n=w.shape[0]
        pre=a.new_empty(m,n);out=torch.empty_like(pre)
        if m:
            bn=32 if n<=32 else 64
            _mm_silu[(tr.cdiv(m,64),tr.cdiv(n,bn))](a,w,bias if bias is not None else out,
                pre,out,m,n,k,bias is not None,64,bn,32,precision,num_warps=4)
        ctx.save_for_backward(pre,w);ctx.shape=shape
        ctx.precision=precision
        return out.reshape(*shape[:-1],n)

    @staticmethod
    def backward(ctx,gradient):
        if torch.is_grad_enabled():raise RuntimeError('Force training needs differentiable reference node linears')
        pre,weight=ctx.saved_tensors
        g=gradient.contiguous();activated=torch.empty_like(pre)
        if pre.numel():
            _silu_gradient[(tr.cdiv(pre.numel(),256),)](pre,g,activated,pre.numel(),256,num_warps=4)
        gradients=(matmul(activated,weight,ctx.precision).reshape(ctx.shape),None,None,None)
        return gradients[:len(ctx.needs_input_grad)]


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,strided_inputs=False,precision='tf32x3'):
        shape=x.shape
        result=(grouped_matmul(x,weight.T,precision) if strided_inputs else
                matmul(x.reshape(-1,shape[-1]).contiguous(),weight.T,precision))
        if bias is not None:result=result+bias
        ctx.save_for_backward(weight)
        ctx.shape=shape
        ctx.strided_inputs=strided_inputs
        ctx.precision=precision
        return result.reshape(*shape[:-1],weight.shape[0])

    @staticmethod
    def backward(ctx,gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("Force training needs ordinary FP32 node linears")
        (weight,)=ctx.saved_tensors
        dx=(grouped_matmul(gradient,weight,ctx.precision) if ctx.strided_inputs else
            matmul(gradient.reshape(-1,weight.shape[0]).contiguous(),weight,ctx.precision))
        gradients=(dx.reshape(ctx.shape),None,None,None,None)
        return gradients[:len(ctx.needs_input_grad)]


class FastLinear(nn.Module):
    def __init__(self,original,precision='tf32x3'):
        super().__init__()
        self.weight,self.bias=original.weight,original.bias
        self.in_features,self.out_features=original.in_features,original.out_features
        self.silu=False
        self.strided_inputs=False
        self.precision=precision
        if precision=='fp16w2':
            quantized=original.weight.detach().to(torch.float16)
            if not torch.equal(quantized.float(),original.weight.detach()):
                raise ValueError('fp16w2 requires explicitly quantized checkpoint weights; it does not silently round a model')
            self.register_buffer('quantized_weight',quantized,persistent=False)

    def forward(self,x):
        if self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad):
            raise RuntimeError("Freeze parameters before enabling fast node inference")
        if min(self.in_features,self.out_features)<16:
            value=F.linear(x,self.weight,self.bias)
            return F.silu(value) if self.silu else value
        weight=self.quantized_weight if self.precision=='fp16w2' else self.weight
        return (_LinearSilu.apply(x,weight,self.bias,self.precision) if self.silu else
                _Linear.apply(x,weight,self.bias,self.strided_inputs,self.precision))


def enable_fast_nodes(model,fuse_activations=False,narrow_fp64=False,strided_inputs=False,precision='tf32x3'):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Fast nodes are an inference backend; freeze parameters first")
    if precision not in ('tf32x3','fp16x3','fp16w2'):
        raise ValueError('Unsupported node multiplication precision')
    def convert(module):
        for name,child in list(module.named_children()):
            if isinstance(child,nn.Linear):
                converted=FastLinear(child,precision);converted.strided_inputs=strided_inputs
                setattr(module,name,converted)
            else:convert(child)
        if fuse_activations and isinstance(module,nn.Sequential):
            for index in range(len(module)-1):
                if isinstance(module[index],FastLinear) and isinstance(module[index+1],nn.SiLU):
                    module[index].silu=True
                    module[index+1]=nn.Identity()
    convert(model)
    model.fast_nodes=True
    model.fused_node_activations=fuse_activations
    model.fast_fp64_readout=narrow_fp64
    model.fast_strided_inputs=strided_inputs
    model.fast_node_precision=precision
    return model
