from typing import Union, Any, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from .prec_grad_maker import PreconditionedGradientMaker, PreconditionedGradientConfig
from ..core import extend
from ..utils import cholesky_inv, has_reduction
from ..operations import OP_SKETCHED_GRAM


__all__ = ['SengGradientMaker', 'SengGradientConfig']

_invalid_data_size = -1


@dataclass
class SengGradientConfig(PreconditionedGradientConfig):
    data_size: int = _invalid_data_size
    damping: float = 1.e-7
    subsample_size: int = None
    sketching_size: int = 256
    truncated_rank: int = 16


@dataclass
class SketchedEmpFisherInfo:
    in_data: Tensor
    out_grads: Tensor
    sub_in_data: Tensor
    sub_out_grads: Tensor
    in_indices: Tensor
    out_indices: Tensor
    gram_inv: Tensor


_supported_modules = (nn.Linear, nn.Conv2d)


class SengGradientMaker(PreconditionedGradientMaker):
    def __init__(self, model: nn.Module, config: SengGradientConfig):
        super().__init__(model, config)
        self.config: SengGradientConfig = config
        assert config.data_size != _invalid_data_size, 'data_size is not set.'
        self._modules = [m for m in model.modules() if isinstance(m, _supported_modules)]
        self._curvature_info: Dict[nn.Module, SketchedEmpFisherInfo] = {}

    def _forward_and_backward(self, *args, **kwargs) -> Union[Tuple[Any, Tensor], Any]:
        if self.do_update_curvature():
            rst = self.update_curvature()
        else:
            rst = self.forward()
            self.backward()
        with torch.no_grad():
            self._loss /= self.config.data_size
        self.precondition()
        return rst

    def _call_loss_fn(self) -> Tensor:
        assert has_reduction(self._loss_fn), 'loss_fn has to have "reduction" option'
        if isinstance(self._loss_fn, nn.Module):
            self._loss_fn.reduction = 'sum'
        else:
            self._loss_fn_kwargs['reduction'] = 'sum'
        args, kwargs = self._get_mapped_loss_fn_args_kwargs()
        return self._loss_fn(*args, **kwargs)

    def update_curvature(self):
        config = self.config
        with extend(self.model, OP_SKETCHED_GRAM,ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.LayerNorm]) as cxt:
            cxt.set_sketching_size(config.sketching_size)
            cxt.set_truncated_rank(config.truncated_rank)
            rst = self.forward()
            self.backward()
            for module in self._modules:
                data, sketches, indices, gram = cxt.sketched_inputs_outgrads_gram(module)
                gram_inv = cholesky_inv(gram.div_(config.data_size), config.damping)
                self._curvature_info[module] = SketchedEmpFisherInfo(*data, *sketches, *indices, gram_inv)
        return rst

    @torch.no_grad()
    def precondition(self):
        data_size = self.config.data_size
        damping = self.config.damping
        for module, info in self._curvature_info.items():
            bias = module.bias is not None and module.bias.requires_grad
            g = maybe_flatten_to_2d(module.weight.grad)  # d_out x d_in
            if bias:
                g = torch.cat([g, module.bias.grad.unsqueeze(-1)], dim=1)  # d_out x (d_in + 1)

            # F = (dI + U'U)
            # F^{-1}g = g/d - U' @ (dI + UU')^{-1} @ Ug/d

            # g <- g/d
            g.div_(data_size).div_(damping)

            # approx Ug
            sub_in_data = maybe_unsqueeze_to_3d(info.sub_in_data)  # n x d_in_sub x r
            sub_out_grads = maybe_unsqueeze_to_3d(info.sub_out_grads)  # n x d_out_sub x r
            in_indices, out_indices = info.in_indices, info.out_indices
            sub_g = g[:, :-1] if bias else g
            ratio = 1
            if out_indices is not None:
                sub_g = torch.index_select(sub_g, dim=0, index=out_indices)  # d_out_sub x d_in
                ratio *= g.shape[0] / sub_g.shape[0]  # d_out / d_out_sub
            if in_indices is not None:
                sub_g = torch.index_select(sub_g, dim=1, index=in_indices)  # d_out_sub x d_in_sub
                ratio *= g.shape[1] / sub_g.shape[1]  # d_in / d_in_sub
            sub_g.mul_(ratio)
            v = torch.einsum('njr,njr->n', torch.einsum('ij,nir->njr', sub_g, sub_out_grads), sub_in_data)  # n

            # approx (dI + UU')^{-1} @ Ug
            v = torch.mv(info.gram_inv, v)  # n

            # approx U' @ (dI + UU')^{-1} @ Ug
            in_data = maybe_unsqueeze_to_3d(info.in_data)  # n x d_in x r
            if bias:
                n, _, r = in_data.shape
                in_data = torch.cat([in_data, in_data.new_ones(n, 1, r)], dim=1)  # n x (d_in+1) x r
            out_grads = maybe_unsqueeze_to_3d(info.out_grads)  # n x d_out x r
            v_abs_sqrt = v.abs().sqrt()  # n
            coeff_in = v.div(v_abs_sqrt.sum())  # n
            coeff_out = v_abs_sqrt  # n
            mat_in = torch.einsum('n,n...->...', coeff_in, in_data)  # d_in x r
            mat_out = torch.einsum('n,n...->...', coeff_out, out_grads)  # d_out x r
            m = mat_out @ mat_in.T  # d_out x {d_in or (d_in+1)}
            m.div_(data_size ** 2)

            # approx g - U' @ (dI + UU')^{-1} @ Ug
            g.sub_(m)
            if bias:
                module.weight.grad.copy_(g[:, :-1].contiguous().view_as(module.weight))  # d_out x d_in
                module.bias.grad.copy_(g[:, -1].flatten())  # d_out
            else:
                module.weight.grad.copy_(g.contiguous().view_as(module.weight))  # d_out x d_in


def maybe_unsqueeze_to_3d(tensor: Tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(-1)
    return tensor


def maybe_flatten_to_2d(tensor: Tensor):
    if tensor.ndim > 2:
        return tensor.flatten(start_dim=1)
    return tensor
