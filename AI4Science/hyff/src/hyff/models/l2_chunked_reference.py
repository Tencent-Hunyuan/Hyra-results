"""Memory-bounded L2 reference using ordinary PyTorch autodiff.

The full system is evaluated; no atoms or neighbors are subsampled. Only edge
execution is chunked. Backward recomputes each chunk with autograd, rather than
using the fused kernel's hand-derived adjoints. This slower implementation is
for checks of large systems before or after MD, outside the timed trajectory.
"""
import torch


def edge_values(displacement, message, vector, tensor, valid, table, radial_filter, rbf, cutoff, width, first):
    from hyff.models.l2_potential import (
        smooth_switch, _quadrupole, _tensor_dot_direction, _vector_tensor_product,
    )
    distance=torch.linalg.vector_norm(displacement,dim=-1).clamp_min(1e-9)
    direction=displacement/distance[:,None]
    quadrupole=_quadrupole(direction)
    radial=table(distance) if table is not None else radial_filter(rbf(distance))
    gate=message*radial*(.25*smooth_switch(distance,cutoff,width)*valid)[:,None]
    if first:
        scalar,vr,tr=gate.chunk(3,-1)
        return scalar,vr[:,None,:]*direction[:,:,None],tr[:,None,:]*quadrupole[:,:,None]
    scalar,vr,v,tv,tr,t,vt=gate.chunk(7,-1)
    tensor_vector,tensor_projection=_tensor_dot_direction(tensor,direction)
    vector_tensor,vector_projection=_vector_tensor_product(vector,direction)
    return (scalar+vt*vector_projection+tv*tensor_projection,
            v[:,None,:]*vector+vr[:,None,:]*direction[:,:,None]+tv[:,None,:]*tensor_vector,
            t[:,None,:]*tensor+tr[:,None,:]*quadrupole[:,:,None]+vt[:,None,:]*vector_tensor)


class _ChunkedReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx,displacement,source,destination,message,vector,tensor,valid,table,radial_filter,rbf,cutoff,width,first,chunk):
        outputs=[torch.zeros_like(message[:,:vector.shape[-1]]),torch.zeros_like(vector),torch.zeros_like(tensor)]
        for start in range(0,len(source),chunk):
            ids=slice(start,start+chunk)
            j=source[ids]
            values=edge_values(displacement[ids],message[j],vector[j],tensor[j],valid[ids],table,radial_filter,rbf,cutoff,width,first)
            for output,value in zip(outputs,values):output.index_add_(0,destination[ids],value)
        ctx.save_for_backward(displacement,source,destination,message,vector,tensor,valid)
        ctx.settings=(table,radial_filter,rbf,cutoff,width,first,chunk)
        return tuple(outputs)

    @staticmethod
    def backward(ctx,gs,gv,gt):
        if torch.is_grad_enabled():raise RuntimeError('The chunked reference is for inference verification, not force training')
        displacement,source,destination,message,vector,tensor,valid=ctx.saved_tensors
        table,radial_filter,rbf,cutoff,width,first,chunk=ctx.settings
        gradients=[torch.zeros_like(message),torch.zeros_like(vector),torch.zeros_like(tensor)]
        gradient_displacement=torch.empty_like(displacement)
        output_gradients=[gs,gv,gt]
        for start in range(0,len(source),chunk):
            ids=slice(start,start+chunk)
            j,i=source[ids],destination[ids]
            with torch.enable_grad():
                inputs=[x.detach().requires_grad_(True) for x in (displacement[ids],message[j],vector[j],tensor[j])]
                outputs=edge_values(*inputs,valid[ids],table,radial_filter,rbf,cutoff,width,first)
                loss=sum((out*gradient[i]).sum() for out,gradient in zip(outputs,output_gradients) if gradient is not None)
                derivatives=torch.autograd.grad(loss,inputs,allow_unused=True)
            gradient_displacement[ids]=derivatives[0]
            for target,value in zip(gradients,derivatives[1:]):
                if value is not None:target.index_add_(0,j,value)
        return gradient_displacement,None,None,*gradients,None,None,None,None,None,None,None,None


def reference_messages(displacement,source,destination,message,vector,tensor,valid,table,radial_filter,rbf,cutoff,width,first,chunk):
    return _ChunkedReference.apply(displacement,source,destination,message,vector,tensor,valid,table,radial_filter,rbf,cutoff,width,first,chunk)
