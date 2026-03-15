import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        block_size=None,
        threshold=1e-6,
        return_ms=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.block_size = block_size
        self.threshold = threshold
        self.return_ms = return_ms

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._last_sparse_ms = 0.0
        self._triton_available = False
        try:
            import triton  # noqa: F401
            self._triton_available = True
        except ImportError:
            pass

        self._w_cl = None
        self._w_cl_version = -1
        self._ag_count_buf = None
        self._ag_list_buf = None

        # 路径缓存
        self._force_zero = False
        self._force_dense = False
        self._fallback_warmup_left = 4

        # 零输出模板缓存
        # key = (device, dtype, H_out, W_out, bias_version)
        self._zero_template_cache = {}

    def _get_w_cl(self):
        ver = self.weight._version
        if self._w_cl is None or self._w_cl_version != ver:
            k = self.kernel_size[0]
            if k == 3:
                self._w_cl = self.weight.data.half().permute(
                    0, 2, 3, 1
                ).contiguous()
            else:
                self._w_cl = self.weight.data.half().reshape(
                    self.out_channels, self.in_channels
                ).contiguous()
            self._w_cl_version = ver
        return self._w_cl

    def _ensure_buffers(self, x):
        from Kernels.conv2d import _select_tile_sizes, GROUP_SIZE
        import triton

        N, C_IN, H, W = x.shape
        BH, BW = _select_tile_sizes(H, W)
        GH = triton.cdiv(H, BH)
        GW = triton.cdiv(W, BW)
        N_TILES = N * GH * GW
        NUM_GROUPS = triton.cdiv(C_IN, GROUP_SIZE)
        MAX_AG = NUM_GROUPS

        if (
            self._ag_count_buf is None
            or self._ag_count_buf.numel() < N_TILES
            or self._ag_count_buf.device != x.device
        ):
            self._ag_count_buf = torch.empty(
                N_TILES, dtype=torch.int32, device=x.device
            )

        needed_list = N_TILES * MAX_AG
        if (
            self._ag_list_buf is None
            or self._ag_list_buf.numel() < needed_list
            or self._ag_list_buf.device != x.device
        ):
            self._ag_list_buf = torch.empty(
                needed_list, dtype=torch.int32, device=x.device
            )

        return self._ag_count_buf, self._ag_list_buf

    def _output_hw(self, H, W):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        H_out = math.floor((H + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
        W_out = math.floor((W + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
        return H_out, W_out

    def _get_zero_template(self, device, dtype, H_out, W_out):
        bias_ver = -1 if self.bias is None else self.bias._version
        key = (str(device), str(dtype), H_out, W_out, bias_ver)

        if key not in self._zero_template_cache:
            template = torch.zeros(
                1, self.out_channels, H_out, W_out,
                dtype=dtype, device=device
            )
            if self.bias is not None:
                template += self.bias.detach().to(dtype=dtype, device=device).view(1, -1, 1, 1)
            self._zero_template_cache[key] = template

        return self._zero_template_cache[key]

    def _zero_output_4d(self, x):
        N, _, H, W = x.shape
        H_out, W_out = self._output_hw(H, W)
        template = self._get_zero_template(
            device=x.device,
            dtype=torch.float32,
            H_out=H_out,
            W_out=W_out,
        )
        self._last_sparse_ms = 0.0
        return template.expand(N, -1, -1, -1)

    def _zero_output_5d(self, x):
        T, B, _, H, W = x.shape
        H_out, W_out = self._output_hw(H, W)
        template = self._get_zero_template(
            device=x.device,
            dtype=torch.float32,
            H_out=H_out,
            W_out=W_out,
        )
        self._last_sparse_ms = 0.0
        return template.expand(T * B, -1, -1, -1).reshape(
            T, B, self.out_channels, H_out, W_out
        )

    @classmethod
    def from_dense(
        cls,
        conv: nn.Conv2d,
        block_size=None,
        threshold: float = 1e-6,
        return_ms: bool = False,
    ) -> "SparseConv2d":
        sparse_conv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            block_size=block_size,
            threshold=threshold,
            return_ms=return_ms,
        )
        sparse_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            sparse_conv.bias.data.copy_(conv.bias.data)
        return sparse_conv.to(conv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 零路径优先，直接绕过 flatten / kernel / buffer
        if self._force_zero:
            if x.dim() == 5:
                return self._zero_output_5d(x)
            elif x.dim() == 4:
                return self._zero_output_4d(x)
            else:
                raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        reshaped = False
        if x.dim() == 5:
            T, B, C, H, W = x.shape
            x = x.reshape(T * B, C, H, W)
            reshaped = True

        use_triton = (
            self._triton_available
            and x.is_cuda
            and self.stride == (1, 1)
            and self.dilation == (1, 1)
            and self.groups == 1
            and self.kernel_size in [(3, 3), (1, 1)]
        )

        if use_triton:
            y = self._triton_forward(x)
        else:
            y = self._fallback_forward(x)

        if reshaped:
            _, C_out, H_out, W_out = y.shape
            y = y.reshape(T, B, C_out, H_out, W_out)

        return y

    def _triton_forward(self, x):
        from Kernels.conv2d import sparse_conv2d_forward

        if self._force_zero:
            return self._zero_output_4d(x)

        if self._force_dense:
            return self._fallback_forward(x)

        w_cl = self._get_w_cl()
        ag_count_buf, ag_list_buf = self._ensure_buffers(x)
        k = self.kernel_size[0]

        if self._fallback_warmup_left > 0:
            y, sparse_ms, avg_active_ratio = sparse_conv2d_forward(
                x=x.contiguous(),
                weight=self.weight,
                bias=self.bias,
                block_size=self.block_size,
                kernel_size=k,
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0],
                groups=self.groups,
                threshold=self.threshold,
                w_cl=w_cl,
                ag_count_buf=ag_count_buf,
                ag_list_buf=ag_list_buf,
                return_ms=self.return_ms,
                return_avg_active_ratio=True,
            )
            self._last_sparse_ms = sparse_ms

            if avg_active_ratio == 0.0:
                self._force_zero = True
            elif avg_active_ratio > 0.85:
                self._force_dense = True

            self._fallback_warmup_left -= 1

            if self._force_zero:
                return self._zero_output_4d(x)
            if self._force_dense:
                return self._fallback_forward(x)
            return y

        y, sparse_ms = sparse_conv2d_forward(
            x=x.contiguous(),
            weight=self.weight,
            bias=self.bias,
            block_size=self.block_size,
            kernel_size=k,
            stride=self.stride[0],
            padding=self.padding[0],
            dilation=self.dilation[0],
            groups=self.groups,
            threshold=self.threshold,
            w_cl=w_cl,
            ag_count_buf=ag_count_buf,
            ag_list_buf=ag_list_buf,
            return_ms=self.return_ms,
            return_avg_active_ratio=False,
        )
        self._last_sparse_ms = sparse_ms
        return y

    def _fallback_forward(self, x):
        self._last_sparse_ms = 0.0
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self):
        bs = self.block_size if self.block_size is not None else "auto"
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, block_size={bs}, "
            f"return_ms={self.return_ms}, sparse=True"
        )