#!/usr/bin/env python
"""Modulated deformable convolution (DCNv2) backed by torchvision.

The original EPicker tree bundled a hand-written CUDA/C++ extension written
against the THC API (``THC/THC.h``, ``THCudaMalloc``,
``THCudaBlas_SgemmBatched``). THC was removed from PyTorch in 1.11, so that
extension cannot be compiled against any modern PyTorch, on any architecture.

``torchvision.ops.deform_conv2d`` implements exactly the same operator
(modulated deformable convolution v2, Zhu et al. 2019) with CPU and CUDA
kernels maintained upstream, so the extension is replaced by this wrapper.
That removes the per-architecture / per-PyTorch-version native build entirely.

Only ``DCN`` is kept: it is the sole symbol EPicker imports
(``lib/models/networks/pose_dla_dcn.py`` and ``resnet_dcn.py``). The old
explicit-offset ``DCNv2`` class and the ``DCNv2Pooling`` / ``DCNPooling``
ROI-pooling variants were referenced only by the extension's own test scripts.
"""

import math
from typing import cast

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d  # torchvision ships no py.typed marker

__all__ = ["DCN"]



def _pair2(value) -> tuple[int, int]:
    """``torch.nn.modules.utils._pair`` narrowed to a concrete 2-tuple of ints."""
    return cast(tuple[int, int], _pair(value))


class DCN(nn.Module):
    """Deformable convolution v2 that predicts its own offsets and mask.

    Drop-in replacement for the CUDA-extension class of the same name.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair2(kernel_size)
        self.stride = _pair2(stride)
        self.padding = _pair2(padding)
        self.dilation = _pair2(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            self.bias.zero_()
            # zero-initialised offsets make the layer start as a plain convolution
            self.conv_offset_mask.weight.zero_()
            if self.conv_offset_mask.bias is not None:
                self.conv_offset_mask.bias.zero_()

    # kept for checkpoint/API compatibility with the original implementation
    init_offset = reset_parameters

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
