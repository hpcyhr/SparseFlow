import math
import torch
import torch.nn as nn


class StaticZeroConv2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        if bias is not None:
            self.register_buffer("bias_buf", bias.detach().float().clone())
        else:
            self.bias_buf = None

        self._template_cache = {}

    @classmethod
    def from_conv(cls, conv: nn.Conv2d):
        return cls(
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=conv.bias,
        ).to(conv.weight.device)

    def _output_hw(self, H, W):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        H_out = math.floor((H + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
        W_out = math.floor((W + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
        return H_out, W_out

    def _get_template(self, device, dtype, H_out, W_out):
        key = (str(device), str(dtype), H_out, W_out)
        if key not in self._template_cache:
            y = torch.zeros(1, self.out_channels, H_out, W_out, dtype=dtype, device=device)
            if self.bias_buf is not None:
                y += self.bias_buf.to(device=device, dtype=dtype).view(1, -1, 1, 1)
            self._template_cache[key] = y
        return self._template_cache[key]

    def forward(self, x):
        if x.dim() == 4:
            B, _, H, W = x.shape
            H_out, W_out = self._output_hw(H, W)
            tpl = self._get_template(x.device, torch.float32, H_out, W_out)
            return tpl.expand(B, -1, -1, -1)

        elif x.dim() == 5:
            T, B, _, H, W = x.shape
            H_out, W_out = self._output_hw(H, W)
            tpl = self._get_template(x.device, torch.float32, H_out, W_out)
            return tpl.expand(T * B, -1, -1, -1).reshape(T, B, self.out_channels, H_out, W_out)

        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")