# """
# SparseFlow Conv2d Triton Kernels — v15.1

# ═══════════════════════════════════════════════════════════════════════
# v14.1 → v15.1: 完整 triton.autotune

#   Autotune 搜索空间（Stage-2 kernel）：
#     BLOCK_M: [64, 128]   — 空间 tile 像素数
#     BLOCK_N: [64, 128]   — 输出通道 tile 大小
#     BLOCK_K: [32, 64]    — K 维 chunk（活跃通道）
#     num_warps: [4, 8]    — warp 并发度
#     num_stages: 1        — 固定（while 循环编译稳定性）
#     → 共 2×2×2×2 = 16 种配置

#   BLOCK_M 与空间 tile 的映射：
#     BLOCK_M=64  → BLOCK_H=8,  BLOCK_W=8  → 每个 tile 覆盖 8×8 像素
#     BLOCK_M=128 → BLOCK_H=8,  BLOCK_W=16 → 每个 tile 覆盖 8×16 像素

#   Prescan 与 Stage-2 的 BH/BW 耦合：
#     Prescan 的 tile 划分由 Python 层的 BH/BW 决定。
#     当 autotune 选择不同的 BLOCK_M 时，BH/BW 也随之变化，
#     因此 Python 层在 autotune 完成首次搜索后，会用选定的
#     BH/BW 来重新运行 Prescan。

#     为简化实现，sparse_conv2d_forward 接受 BH/BW 参数，
#     由 SparseConv2d 模块在首次运行确定后缓存。
#     对于 bench_resnet.py 的直接调用，默认 BH/BW 由
#     _select_tile_sizes 自动选择。

#   key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW']：
#     不同层维度和 tile 配置独立调优缓存。

#   架构不变：
#   - Per-Tile 通道稀疏（感受野扩展 PAD 像素）
#   - Channel-Last 权重 [C_OUT, 3, 3, C_IN]
#   - 间接索引（不在 Python 层 gather weight）
#   - while 循环动态跳过
#   - 零 Host-Device 同步
# ═══════════════════════════════════════════════════════════════════════
# """

# import torch
# import triton
# import triton.language as tl
# from triton import autotune, Config


# # ═══════════════════════════════════════════════════════════════════════
# # Tile 大小选择（确定 Prescan 的 BH/BW，也决定 Stage-2 的 Grid）
# # ═══════════════════════════════════════════════════════════════════════

# def _select_tile_sizes(H, W):
#     """
#     选择空间 Tile 尺寸 BH/BW。

#     H>=56: BH=8, BW=16 → BLOCK_M=128, GH=7, GW=4 (28 tiles/sample)
#     else:  BH=8, BW=8  → BLOCK_M=64
#     """
#     pixels = H * W
#     if pixels >= 3136:      # H≈56
#         return 8, 16
#     else:
#         return 8, 8


# # 向后兼容接口
# def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
#     BH, BW = _select_tile_sizes(H, W)
#     return BH, BW, BH * BW, 64, 32


# # ═══════════════════════════════════════════════════════════════════════
# # Stage-1 Step 1: 统计每个 Tile 感受野内的活跃通道数
# # ═══════════════════════════════════════════════════════════════════════

# @triton.jit
# def prescan_count_kernel(
#     x_ptr, counts_ptr,
#     N_val, C_IN, H, W, GH, GW,
#     BLOCK_H: tl.constexpr,
#     BLOCK_W: tl.constexpr,
#     PAD: tl.constexpr,
#     MAX_C: tl.constexpr,
#     RF_SIZE: tl.constexpr,
#     THRESHOLD: tl.constexpr,
# ):
#     """
#     对每个输出 Tile，统计其感受野内的活跃输入通道数。
#     感受野 = (BH+2*PAD) × (BW+2*PAD)，展平后 pad 到 RF_SIZE (power-of-2)。
#     Grid: (N_TILES,)
#     """
#     tile_id = tl.program_id(0)
#     total_tiles = N_val * GH * GW
#     if tile_id >= total_tiles:
#         return

#     gw_idx = tile_id % GW
#     tmp = tile_id // GW
#     gh_idx = tmp % GH
#     n_idx = tmp // GH

#     rf_h_start = gh_idx * BLOCK_H - PAD
#     rf_w_start = gw_idx * BLOCK_W - PAD
#     RF_H: tl.constexpr = BLOCK_H + 2 * PAD
#     RF_W: tl.constexpr = BLOCK_W + 2 * PAD

#     flat_idx = tl.arange(0, RF_SIZE)
#     flat_row = flat_idx // RF_W
#     flat_col = flat_idx % RF_W
#     valid_mask = flat_idx < (RF_H * RF_W)

#     hh = rf_h_start + flat_row
#     ww = rf_w_start + flat_col
#     hw_mask = valid_mask & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
#     safe_h = tl.minimum(tl.maximum(hh, 0), H - 1)
#     safe_w = tl.minimum(tl.maximum(ww, 0), W - 1)

#     HW = H * W
#     count = 0

#     for c_idx in range(MAX_C):
#         if c_idx < C_IN:
#             base = (n_idx * C_IN + c_idx) * HW
#             vals = tl.load(x_ptr + base + safe_h * W + safe_w,
#                            mask=hw_mask, other=0.0)
#             is_nz = tl.max(tl.abs(vals)) > THRESHOLD
#             count += is_nz.to(tl.int32)

#     tl.store(counts_ptr + tile_id, count)


# # ═══════════════════════════════════════════════════════════════════════
# # Stage-1 Step 3: 写入每个 Tile 的活跃通道索引
# # ═══════════════════════════════════════════════════════════════════════

# @triton.jit
# def prescan_write_kernel(
#     x_ptr, tile_ptr_data, tile_cin_ptr,
#     cin_buf_size,
#     N_val, C_IN, H, W, GH, GW,
#     BLOCK_H: tl.constexpr,
#     BLOCK_W: tl.constexpr,
#     PAD: tl.constexpr,
#     MAX_C: tl.constexpr,
#     RF_SIZE: tl.constexpr,
#     THRESHOLD: tl.constexpr,
# ):
#     """
#     对每个 Tile，将其感受野内的活跃 Cin 索引写入 tile_cin_buf。
#     Grid: (N_TILES,)
#     """
#     tile_id = tl.program_id(0)
#     total_tiles = N_val * GH * GW
#     if tile_id >= total_tiles:
#         return

#     gw_idx = tile_id % GW
#     tmp = tile_id // GW
#     gh_idx = tmp % GH
#     n_idx = tmp // GH

#     rf_h_start = gh_idx * BLOCK_H - PAD
#     rf_w_start = gw_idx * BLOCK_W - PAD
#     RF_H: tl.constexpr = BLOCK_H + 2 * PAD
#     RF_W: tl.constexpr = BLOCK_W + 2 * PAD

#     flat_idx = tl.arange(0, RF_SIZE)
#     flat_row = flat_idx // RF_W
#     flat_col = flat_idx % RF_W
#     valid_mask = flat_idx < (RF_H * RF_W)

#     hh = rf_h_start + flat_row
#     ww = rf_w_start + flat_col
#     hw_mask = valid_mask & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
#     safe_h = tl.minimum(tl.maximum(hh, 0), H - 1)
#     safe_w = tl.minimum(tl.maximum(ww, 0), W - 1)

#     HW = H * W
#     write_pos = tl.load(tile_ptr_data + tile_id)
#     idx = 0

#     for c_idx in range(MAX_C):
#         if c_idx < C_IN:
#             base = (n_idx * C_IN + c_idx) * HW
#             vals = tl.load(x_ptr + base + safe_h * W + safe_w,
#                            mask=hw_mask, other=0.0)
#             is_nz = tl.max(tl.abs(vals)) > THRESHOLD

#             if is_nz:
#                 out_pos = write_pos + idx
#                 if out_pos < cin_buf_size:
#                     tl.store(tile_cin_ptr + out_pos, c_idx)
#                 idx += 1


# # ═══════════════════════════════════════════════════════════════════════
# # Stage-1 Python 编排：count → cumsum → write（零同步版）
# # ═══════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def _build_tile_csr(x_f16, N, C_IN, H, W, BH, BW, GH, GW,
#                     kernel_size, threshold,
#                     counts_buf, tile_cin_buf):
#     """
#     全 GPU 构建 Per-Tile CSR 结构。零 Host-Device 同步。

#     Returns:
#         tile_ptr: [N_TILES + 1] int32
#     """
#     device = x_f16.device
#     N_TILES = N * GH * GW
#     PAD = 1 if kernel_size == 3 else 0
#     MAX_C = triton.next_power_of_2(max(C_IN, 1))
#     rf_actual = (BH + 2 * PAD) * (BW + 2 * PAD)
#     RF_SIZE = triton.next_power_of_2(rf_actual)

#     tile_counts = counts_buf[:N_TILES]

#     prescan_count_kernel[(N_TILES,)](
#         x_f16, tile_counts,
#         N, C_IN, H, W, GH, GW,
#         BLOCK_H=BH, BLOCK_W=BW,
#         PAD=PAD, MAX_C=MAX_C,
#         RF_SIZE=RF_SIZE,
#         THRESHOLD=threshold,
#     )

#     cumsum = torch.cumsum(tile_counts, dim=0, dtype=torch.int32)
#     tile_ptr = torch.empty(N_TILES + 1, dtype=torch.int32, device=device)
#     tile_ptr[0] = 0
#     tile_ptr[1:] = cumsum

#     cin_buf_size = tile_cin_buf.numel()

#     prescan_write_kernel[(N_TILES,)](
#         x_f16, tile_ptr, tile_cin_buf,
#         cin_buf_size,
#         N, C_IN, H, W, GH, GW,
#         BLOCK_H=BH, BLOCK_W=BW,
#         PAD=PAD, MAX_C=MAX_C,
#         RF_SIZE=RF_SIZE,
#         THRESHOLD=threshold,
#     )

#     return tile_ptr


# # ═══════════════════════════════════════════════════════════════════════
# # Autotune 配置池 — 16 种组合
# # ═══════════════════════════════════════════════════════════════════════

# # BLOCK_M=64  → BLOCK_H=8, BLOCK_W=8
# # BLOCK_M=128 → BLOCK_H=8, BLOCK_W=16
# # 映射关系硬编码在 Config 的 kwargs 中

# _CONV3X3_CONFIGS = [
#     # ── BLOCK_M=64 (8×8 tile) ──
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     # ── BLOCK_M=128 (8×16 tile) ──
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
# ]

# _CONV1X1_CONFIGS = [
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
#     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
#             'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
# ]


# # ═══════════════════════════════════════════════════════════════════════
# # Stage-2: Per-Tile 稀疏 3×3 Conv（autotune + while 动态跳过）
# # ═══════════════════════════════════════════════════════════════════════

# @autotune(configs=_CONV3X3_CONFIGS,
#           key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
# @triton.jit
# def sparse_conv3x3_pertile_kernel(
#     x_ptr,              # [N, C_IN, H, W] fp16
#     w_cl_ptr,           # [C_OUT, 3, 3, C_IN] fp16 (Channel-Last)
#     bias_ptr,           # [C_OUT] fp32 or dummy
#     tile_ptr_data,      # [N_TILES + 1] int32
#     tile_cin_ptr,       # [max_active_entries] int32
#     y_ptr,              # [N, C_OUT, H, W] fp32
#     N_val,
#     C_IN: tl.constexpr,
#     C_OUT: tl.constexpr,
#     H: tl.constexpr,
#     W: tl.constexpr,
#     GH: tl.constexpr,
#     GW: tl.constexpr,
#     HAS_BIAS: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     BLOCK_H: tl.constexpr,
#     BLOCK_W: tl.constexpr,
# ):
#     """
#     Per-Tile 稀疏 3×3 卷积 — autotune + while 动态跳过。

#     Autotuner 搜索 BLOCK_M/N/K 和 num_warps 的最优组合。
#     key=(C_IN, C_OUT, H, W, GH, GW) 确保不同层和 tile 划分独立调优。
#     Grid 由 lambda META 动态计算 N_TILES 和 grid_cout。
#     """
#     tile_id = tl.program_id(0)
#     pid_cout = tl.program_id(1)

#     total_tiles = N_val * GH * GW
#     if tile_id >= total_tiles:
#         return

#     gw_idx = tile_id % GW
#     tmp = tile_id // GW
#     gh_idx = tmp % GH
#     n_idx = tmp // GH

#     c_out_start = pid_cout * BLOCK_N
#     offs_n = c_out_start + tl.arange(0, BLOCK_N)
#     n_mask = offs_n < C_OUT

#     offs_m = tl.arange(0, BLOCK_M)
#     tile_bh = offs_m // BLOCK_W
#     tile_bw = offs_m % BLOCK_W
#     out_h = gh_idx * BLOCK_H + tile_bh
#     out_w = gw_idx * BLOCK_W + tile_bw
#     m_mask = (out_h < H) & (out_w < W)

#     HW: tl.constexpr = H * W

#     tile_start = tl.load(tile_ptr_data + tile_id)
#     tile_end = tl.load(tile_ptr_data + tile_id + 1)
#     active_K = tile_end - tile_start

#     acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

#     # 权重布局 [C_OUT, 3, 3, C_IN] 的步长常量
#     W_CIN_STRIDE: tl.constexpr = C_IN
#     W_KH_STRIDE: tl.constexpr = 3 * C_IN
#     W_CO_STRIDE: tl.constexpr = 9 * C_IN

#     # ── while 循环：真正的动态 K 迭代 ──
#     k_start = 0
#     while k_start < active_K:
#         offs_k = k_start + tl.arange(0, BLOCK_K)
#         k_mask = offs_k < active_K

#         cin_global = tl.load(
#             tile_cin_ptr + tile_start + offs_k,
#             mask=k_mask, other=0)

#         for kh in tl.static_range(3):
#             for kw in tl.static_range(3):
#                 in_h = out_h + (kh - 1)
#                 in_w = out_w + (kw - 1)
#                 h_ok = (in_h >= 0) & (in_h < H)
#                 w_ok = (in_w >= 0) & (in_w < W)
#                 safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
#                 safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)

#                 x_addrs = (x_ptr
#                            + (n_idx * C_IN + cin_global[None, :]) * HW
#                            + safe_h[:, None] * W
#                            + safe_w[:, None])
#                 x_load_mask = (k_mask[None, :] & m_mask[:, None]
#                                & h_ok[:, None] & w_ok[:, None])
#                 x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
#                 x_tile = x_tile.to(tl.float16)

#                 w_addrs = (w_cl_ptr
#                            + offs_n[None, :] * W_CO_STRIDE
#                            + kh * W_KH_STRIDE
#                            + kw * W_CIN_STRIDE
#                            + cin_global[:, None])
#                 w_load_mask = k_mask[:, None] & n_mask[None, :]
#                 w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0)
#                 w_tile = w_tile.to(tl.float16)

#                 acc += tl.dot(x_tile, w_tile)

#         k_start += BLOCK_K

#     if HAS_BIAS:
#         bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
#         acc += bias_vals[None, :]

#     out_addrs = (y_ptr
#                  + (n_idx * C_OUT + offs_n[None, :]) * HW
#                  + out_h[:, None] * W
#                  + out_w[:, None])
#     out_mask = m_mask[:, None] & n_mask[None, :]
#     tl.store(out_addrs, acc, mask=out_mask)


# # ═══════════════════════════════════════════════════════════════════════
# # Stage-2: Per-Tile 稀疏 1×1 Conv（autotune + while 动态跳过）
# # ═══════════════════════════════════════════════════════════════════════

# @autotune(configs=_CONV1X1_CONFIGS,
#           key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
# @triton.jit
# def sparse_conv1x1_pertile_kernel(
#     x_ptr,              # [N, C_IN, H, W] fp16
#     w_ptr,              # [C_OUT, C_IN] fp16
#     bias_ptr,           # [C_OUT] fp32 or dummy
#     tile_ptr_data,      # [N_TILES + 1] int32
#     tile_cin_ptr,       # [max_active_entries] int32
#     y_ptr,              # [N, C_OUT, H, W] fp32
#     N_val,
#     C_IN: tl.constexpr,
#     C_OUT: tl.constexpr,
#     H: tl.constexpr,
#     W: tl.constexpr,
#     GH: tl.constexpr,
#     GW: tl.constexpr,
#     HAS_BIAS: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     BLOCK_H: tl.constexpr,
#     BLOCK_W: tl.constexpr,
# ):
#     """Per-Tile 稀疏 1×1 conv — autotune + while 动态跳过。"""
#     tile_id = tl.program_id(0)
#     pid_cout = tl.program_id(1)

#     total_tiles = N_val * GH * GW
#     if tile_id >= total_tiles:
#         return

#     gw_idx = tile_id % GW
#     tmp = tile_id // GW
#     gh_idx = tmp % GH
#     n_idx = tmp // GH

#     c_out_start = pid_cout * BLOCK_N
#     offs_n = c_out_start + tl.arange(0, BLOCK_N)
#     n_mask = offs_n < C_OUT

#     offs_m = tl.arange(0, BLOCK_M)
#     tile_bh = offs_m // BLOCK_W
#     tile_bw = offs_m % BLOCK_W
#     out_h = gh_idx * BLOCK_H + tile_bh
#     out_w = gw_idx * BLOCK_W + tile_bw
#     m_mask = (out_h < H) & (out_w < W)
#     safe_h = tl.minimum(out_h, H - 1)
#     safe_w = tl.minimum(out_w, W - 1)

#     HW: tl.constexpr = H * W

#     tile_start = tl.load(tile_ptr_data + tile_id)
#     tile_end = tl.load(tile_ptr_data + tile_id + 1)
#     active_K = tile_end - tile_start

#     acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

#     k_start = 0
#     while k_start < active_K:
#         offs_k = k_start + tl.arange(0, BLOCK_K)
#         k_mask = offs_k < active_K

#         cin_global = tl.load(
#             tile_cin_ptr + tile_start + offs_k,
#             mask=k_mask, other=0)

#         x_addrs = (x_ptr
#                    + (n_idx * C_IN + cin_global[None, :]) * HW
#                    + safe_h[:, None] * W
#                    + safe_w[:, None])
#         x_load_mask = k_mask[None, :] & m_mask[:, None]
#         x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0)
#         x_tile = x_tile.to(tl.float16)

#         w_addrs = (w_ptr
#                    + offs_n[None, :] * C_IN
#                    + cin_global[:, None])
#         w_load_mask = k_mask[:, None] & n_mask[None, :]
#         w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0)
#         w_tile = w_tile.to(tl.float16)

#         acc += tl.dot(x_tile, w_tile)
#         k_start += BLOCK_K

#     if HAS_BIAS:
#         bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
#         acc += bias_vals[None, :]

#     out_addrs = (y_ptr
#                  + (n_idx * C_OUT + offs_n[None, :]) * HW
#                  + out_h[:, None] * W
#                  + out_w[:, None])
#     out_mask = m_mask[:, None] & n_mask[None, :]
#     tl.store(out_addrs, acc, mask=out_mask)


# # ═══════════════════════════════════════════════════════════════════════
# # Legacy kernels — 向后兼容
# # ═══════════════════════════════════════════════════════════════════════

# @triton.jit
# def prescan_kernel(
#     x_ptr, flags_ptr, N, C, H, W, GRID_H, GRID_W,
#     BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
#     THRESHOLD: tl.constexpr,
# ):
#     """Legacy per-block prescan."""
#     pid = tl.program_id(0)
#     gw = pid % GRID_W; tmp = pid // GRID_W
#     gh = tmp % GRID_H; tmp2 = tmp // GRID_H
#     c = tmp2 % C; n = tmp2 // C
#     hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
#     ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
#     mask = (hh < H) & (ww < W)
#     base = (n * C + c) * H
#     val = tl.load(x_ptr + (base + hh) * W + ww, mask=mask, other=0.0)
#     is_nz = tl.max(tl.abs(val)) > THRESHOLD
#     tl.store(flags_ptr + pid, is_nz.to(tl.int32))


# @triton.jit
# def dense_conv3x3_kernel(
#     x_ptr, y_ptr, N, C, H, W, GRID_H, GRID_W,
#     BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
# ):
#     """Dense 3×3 box filter baseline."""
#     pid = tl.program_id(0)
#     total = N * C * GRID_H * GRID_W
#     if pid >= total:
#         return
#     gw = pid % GRID_W; tmp = pid // GRID_W
#     gh = tmp % GRID_H; tmp2 = tmp // GRID_H
#     c = tmp2 % C; n = tmp2 // C
#     hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
#     ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
#     mask_hw = (hh < H) & (ww < W)
#     acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
#     for kh in range(-1, 2):
#         for kw in range(-1, 2):
#             h_idx = hh + kh; w_idx = ww + kw
#             m = mask_hw & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
#             acc += tl.load(x_ptr + ((n*C+c)*H+h_idx)*W+w_idx, mask=m, other=0.0)
#     tl.store(y_ptr + ((n*C+c)*H+hh)*W+ww, acc, mask=mask_hw)


# # ═══════════════════════════════════════════════════════════════════════
# # 主入口: sparse_conv2d_forward
# # ═══════════════════════════════════════════════════════════════════════

# def sparse_conv2d_forward(x, weight, bias, block_size=None,
#                           kernel_size=3, threshold=1e-6,
#                           w_cl=None, counts_buf=None, tile_cin_buf=None,
#                           return_ms=False):
#     """
#     Per-Tile 通道稀疏 Conv2d — autotune + 零 Host-Device 同步版。

#     Pipeline:
#       1. prescan_count → cumsum → prescan_write  (全 GPU, 零 sync)
#       2. Stage-2 autotuned sparse GEMM (while 动态跳过, 零 sync)

#     Autotuner 搜索 BLOCK_M/N/K 和 num_warps 的 16 种组合。
#     BH/BW 由 _select_tile_sizes 决定，Prescan 和 Stage-2 共用。
#     Grid 通过 lambda META 动态计算。

#     注意：由于 autotune 可能搜索不同的 BLOCK_M（对应不同的 BH/BW），
#     但 Prescan 必须在 kernel 启动前完成，所以 Prescan 用 Python 层
#     固定的 BH/BW。Autotuner 在 benchmark 各 config 时，如果 config 的
#     BLOCK_H/BLOCK_W 与 Prescan 的 BH/BW 不同，kernel 内部的 tile
#     映射会覆盖不同的像素区域，但 m_mask 保证不越界写入。

#     实际行为：autotuner 的 BLOCK_H/BLOCK_W 来自 Config，与 Prescan 的
#     BH/BW 一致时性能最优。不一致时结果仍然正确（m_mask 保护），
#     但部分 tile 的 CSR 不精确（prescan 用不同粒度扫描）。
#     autotuner 的 benchmark 会自动惩罚这种配置（它更慢），所以最终
#     选出的配置自然倾向于与 Prescan 一致的 BH/BW。

#     Args:
#         x: [N, C_IN, H, W] 输入
#         weight: [C_OUT, C_IN, K, K] 原始权重
#         bias: [C_OUT] or None
#         w_cl: 预转换 Channel-Last 权重
#         counts_buf, tile_cin_buf: 预分配 buffer
#         return_ms: True → CUDA event 计时

#     Returns:
#         y: [N, C_OUT, H, W] fp32
#         sparse_ms: float
#     """
#     N, C_IN, H, W = x.shape
#     C_OUT = weight.shape[0]
#     device = x.device

#     # ── Tile 尺寸 ──
#     BH, BW = _select_tile_sizes(H, W)
#     GH = triton.cdiv(H, BH)
#     GW = triton.cdiv(W, BW)
#     N_TILES = N * GH * GW

#     x_f16 = x.half().contiguous()

#     # ── 权重 Channel-Last 转换 ──
#     if w_cl is not None:
#         w_cl_f16 = w_cl
#     else:
#         if kernel_size == 3:
#             w_cl_f16 = weight.half().permute(0, 2, 3, 1).contiguous()
#         else:
#             w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

#     # ── Buffer 准备 ──
#     if counts_buf is None or counts_buf.numel() < N_TILES:
#         counts_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
#     if tile_cin_buf is None or tile_cin_buf.numel() < N_TILES * C_IN:
#         tile_cin_buf = torch.empty(N_TILES * C_IN, dtype=torch.int32,
#                                    device=device)

#     # ── Stage-1: Per-Tile CSR（零同步） ──
#     tile_ptr = _build_tile_csr(
#         x_f16, N, C_IN, H, W, BH, BW, GH, GW,
#         kernel_size, threshold,
#         counts_buf=counts_buf, tile_cin_buf=tile_cin_buf)

#     # ── Stage-2: Autotuned 稀疏 Conv ──
#     has_bias = bias is not None
#     bias_f32 = (bias.float().contiguous() if has_bias
#                 else torch.empty(1, device=device))
#     y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

#     sparse_ms = 0.0
#     if return_ms:
#         start_evt = torch.cuda.Event(enable_timing=True)
#         end_evt = torch.cuda.Event(enable_timing=True)
#         start_evt.record()

#     # Grid lambda: autotuner 注入 META，从中读取 BLOCK_N 来计算 grid
#     def _grid(META):
#         return (N_TILES, triton.cdiv(C_OUT, META['BLOCK_N']))

#     if kernel_size == 3:
#         sparse_conv3x3_pertile_kernel[_grid](
#             x_f16, w_cl_f16, bias_f32,
#             tile_ptr, tile_cin_buf,
#             y,
#             N,                     # N_val (运行时)
#             C_IN, C_OUT, H, W,    # autotune key (constexpr)
#             GH, GW,               # autotune key (constexpr)
#             HAS_BIAS=has_bias,
#             # BLOCK_M/N/K/H/W 全部由 autotuner 从 Config 注入
#         )
#     else:
#         sparse_conv1x1_pertile_kernel[_grid](
#             x_f16, w_cl_f16, bias_f32,
#             tile_ptr, tile_cin_buf,
#             y,
#             N,
#             C_IN, C_OUT, H, W,
#             GH, GW,
#             HAS_BIAS=has_bias,
#         )

#     if return_ms:
#         end_evt.record()
#         torch.cuda.synchronize(device)
#         sparse_ms = start_evt.elapsed_time(end_evt)

#     return y, sparse_ms

"""
SparseFlow Conv2d Triton Kernels — v15.1 (fixed)

Fix:
  Prescan BH/BW must match Stage-2 kernel BLOCK_H/W exactly.
  We split Stage-2 kernels by tile shape (8x8 vs 8x16) and autotune separately.
"""

import torch
import triton
import triton.language as tl
from triton import autotune, Config


# ═══════════════════════════════════════════════════════════════════════
# Tile 大小选择（确定 Prescan 的 BH/BW，也决定 Stage-2 的 Grid）
# ═══════════════════════════════════════════════════════════════════════

def _select_tile_sizes(H, W):
    """
    选择空间 Tile 尺寸 BH/BW。

    H>=56: BH=8, BW=16 → BLOCK_M=128, GH=7, GW=4 (28 tiles/sample)
    else:  BH=8, BW=8  → BLOCK_M=64
    """
    pixels = H * W
    if pixels >= 3136:      # H≈56
        return 8, 16
    else:
        return 8, 8


# 向后兼容接口
def _select_block_sizes(H, W, C_IN, C_OUT, kernel_size, N):
    BH, BW = _select_tile_sizes(H, W)
    return BH, BW, BH * BW, 64, 32


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Step 1: 统计每个 Tile 感受野内的活跃通道数
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_count_kernel(
    x_ptr, counts_ptr,
    N_val, C_IN, H, W, GH, GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    PAD: tl.constexpr,
    MAX_C: tl.constexpr,
    RF_SIZE: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    """
    对每个输出 Tile，统计其感受野内的活跃输入通道数。
    感受野 = (BH+2*PAD) × (BW+2*PAD)，展平后 pad 到 RF_SIZE (power-of-2)。
    Grid: (N_TILES,)
    """
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    rf_h_start = gh_idx * BLOCK_H - PAD
    rf_w_start = gw_idx * BLOCK_W - PAD
    RF_H: tl.constexpr = BLOCK_H + 2 * PAD
    RF_W: tl.constexpr = BLOCK_W + 2 * PAD

    flat_idx = tl.arange(0, RF_SIZE)
    flat_row = flat_idx // RF_W
    flat_col = flat_idx % RF_W
    valid_mask = flat_idx < (RF_H * RF_W)

    hh = rf_h_start + flat_row
    ww = rf_w_start + flat_col
    hw_mask = valid_mask & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
    safe_h = tl.minimum(tl.maximum(hh, 0), H - 1)
    safe_w = tl.minimum(tl.maximum(ww, 0), W - 1)

    HW = H * W

    # ✅ Triton in your env forbids 0-d blocks
    count = tl.zeros([1], dtype=tl.int32)

    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            base = (n_idx * C_IN + c_idx) * HW
            vals = tl.load(
                x_ptr + base + safe_h * W + safe_w,
                mask=hw_mask, other=0.0
            )
            is_nz_i32 = (tl.max(tl.abs(vals)) > THRESHOLD).to(tl.int32)
            # make it a 1-element block to match `count`
            count += tl.full([1], is_nz_i32, tl.int32)

    # ✅ store 1 element
    off1 = tl.arange(0, 1)
    tl.store(counts_ptr + tile_id + off1, count)


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Step 3: 写入每个 Tile 的活跃通道索引
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_write_kernel(
    x_ptr, tile_ptr_data, tile_cin_ptr,
    cin_buf_size,
    N_val, C_IN, H, W, GH, GW,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    PAD: tl.constexpr,
    MAX_C: tl.constexpr,
    RF_SIZE: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    tile_id = tl.program_id(0)
    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    rf_h_start = gh_idx * BLOCK_H - PAD
    rf_w_start = gw_idx * BLOCK_W - PAD
    RF_H: tl.constexpr = BLOCK_H + 2 * PAD
    RF_W: tl.constexpr = BLOCK_W + 2 * PAD

    flat_idx = tl.arange(0, RF_SIZE)
    flat_row = flat_idx // RF_W
    flat_col = flat_idx % RF_W
    valid_mask = flat_idx < (RF_H * RF_W)

    hh = rf_h_start + flat_row
    ww = rf_w_start + flat_col
    hw_mask = valid_mask & (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
    safe_h = tl.minimum(tl.maximum(hh, 0), H - 1)
    safe_w = tl.minimum(tl.maximum(ww, 0), W - 1)

    HW = H * W

    off1 = tl.arange(0, 1)
    write_pos = tl.load(tile_ptr_data + tile_id + off1)  # [1]
    idx = tl.zeros([1], dtype=tl.int32)                  # [1]

    for c_idx in range(MAX_C):
        if c_idx < C_IN:
            base = (n_idx * C_IN + c_idx) * HW
            vals = tl.load(
                x_ptr + base + safe_h * W + safe_w,
                mask=hw_mask, other=0.0
            )
            is_nz = tl.max(tl.abs(vals)) > THRESHOLD

            if is_nz:
                out_pos = write_pos + idx  # [1]
                can_store = out_pos < cin_buf_size
                tl.store(tile_cin_ptr + out_pos, c_idx, mask=can_store)
                idx += 1


# ═══════════════════════════════════════════════════════════════════════
# Stage-1 Python 编排：count → cumsum → write（零同步版）
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_tile_csr(x_f16, N, C_IN, H, W, BH, BW, GH, GW,
                    kernel_size, threshold,
                    counts_buf, tile_cin_buf):
    """
    全 GPU 构建 Per-Tile CSR 结构。零 Host-Device 同步。

    Returns:
        tile_ptr: [N_TILES + 1] int32
    """
    device = x_f16.device
    N_TILES = N * GH * GW
    PAD = 1 if kernel_size == 3 else 0
    MAX_C = triton.next_power_of_2(max(C_IN, 1))
    rf_actual = (BH + 2 * PAD) * (BW + 2 * PAD)
    RF_SIZE = triton.next_power_of_2(rf_actual)

    tile_counts = counts_buf[:N_TILES]

    prescan_count_kernel[(N_TILES,)](
        x_f16, tile_counts,
        N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW,
        PAD=PAD, MAX_C=MAX_C,
        RF_SIZE=RF_SIZE,
        THRESHOLD=threshold,
    )

    cumsum = torch.cumsum(tile_counts, dim=0, dtype=torch.int32)
    tile_ptr = torch.empty(N_TILES + 1, dtype=torch.int32, device=device)
    tile_ptr[0] = 0
    tile_ptr[1:] = cumsum

    cin_buf_size = tile_cin_buf.numel()

    prescan_write_kernel[(N_TILES,)](
        x_f16, tile_ptr, tile_cin_buf,
        cin_buf_size,
        N, C_IN, H, W, GH, GW,
        BLOCK_H=BH, BLOCK_W=BW,
        PAD=PAD, MAX_C=MAX_C,
        RF_SIZE=RF_SIZE,
        THRESHOLD=threshold,
    )

    return tile_ptr


# ═══════════════════════════════════════════════════════════════════════
# Autotune 配置池 — 16 种组合（按 tile 形状拆分）
# ═══════════════════════════════════════════════════════════════════════

_CONV3X3_CONFIGS = [
    # ── BLOCK_M=64 (8×8 tile) ──
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),

    # ── BLOCK_M=128 (8×16 tile) ──
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
]

_CONV3X3_CONFIGS_8x8  = [cfg for cfg in _CONV3X3_CONFIGS if cfg.kwargs.get('BLOCK_W') == 8]
_CONV3X3_CONFIGS_8x16 = [cfg for cfg in _CONV3X3_CONFIGS if cfg.kwargs.get('BLOCK_W') == 16]

_CONV1X1_CONFIGS = [
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=4, num_stages=1),
    Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 8},  num_warps=8, num_stages=1),

    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=1),
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
            'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=1),
]

_CONV1X1_CONFIGS_8x8  = [cfg for cfg in _CONV1X1_CONFIGS if cfg.kwargs.get('BLOCK_W') == 8]
_CONV1X1_CONFIGS_8x16 = [cfg for cfg in _CONV1X1_CONFIGS if cfg.kwargs.get('BLOCK_W') == 16]


# ═══════════════════════════════════════════════════════════════════════
# Stage-2: Per-Tile 稀疏 3×3 Conv（按 tile 形状拆分 autotune）
# ═══════════════════════════════════════════════════════════════════════

@autotune(configs=_CONV3X3_CONFIGS_8x8,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def sparse_conv3x3_pertile_kernel_8x8(
    x_ptr,              # [N, C_IN, H, W] fp16
    w_cl_ptr,           # [C_OUT, 3, 3, C_IN] fp16 (Channel-Last)
    bias_ptr,           # [C_OUT] fp32 or dummy
    tile_ptr_data,      # [N_TILES + 1] int32
    tile_cin_ptr,       # [max_active_entries] int32
    y_ptr,              # [N, C_OUT, H, W] fp32
    N_val,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GH: tl.constexpr,
    GW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    out_h = gh_idx * BLOCK_H + tile_bh
    out_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (out_h < H) & (out_w < W)

    HW: tl.constexpr = H * W

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    W_CIN_STRIDE: tl.constexpr = C_IN
    W_KH_STRIDE: tl.constexpr = 3 * C_IN
    W_CO_STRIDE: tl.constexpr = 9 * C_IN

    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H)
                w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)

                x_addrs = (x_ptr
                           + (n_idx * C_IN + cin_global[None, :]) * HW
                           + safe_h[:, None] * W
                           + safe_w[:, None])
                x_load_mask = (k_mask[None, :] & m_mask[:, None]
                               & h_ok[:, None] & w_ok[:, None])
                x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0).to(tl.float16)

                w_addrs = (w_cl_ptr
                           + offs_n[None, :] * W_CO_STRIDE
                           + kh * W_KH_STRIDE
                           + kw * W_CIN_STRIDE
                           + cin_global[:, None])
                w_load_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0).to(tl.float16)

                acc += tl.dot(x_tile, w_tile)

        k_start += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + out_h[:, None] * W
                 + out_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


@autotune(configs=_CONV3X3_CONFIGS_8x16,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def sparse_conv3x3_pertile_kernel_8x16(
    x_ptr, w_cl_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # same body as 8x8 variant
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    out_h = gh_idx * BLOCK_H + tile_bh
    out_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (out_h < H) & (out_w < W)

    HW: tl.constexpr = H * W

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    W_CIN_STRIDE: tl.constexpr = C_IN
    W_KH_STRIDE: tl.constexpr = 3 * C_IN
    W_CO_STRIDE: tl.constexpr = 9 * C_IN

    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                in_h = out_h + (kh - 1)
                in_w = out_w + (kw - 1)
                h_ok = (in_h >= 0) & (in_h < H)
                w_ok = (in_w >= 0) & (in_w < W)
                safe_h = tl.minimum(tl.maximum(in_h, 0), H - 1)
                safe_w = tl.minimum(tl.maximum(in_w, 0), W - 1)

                x_addrs = (x_ptr
                           + (n_idx * C_IN + cin_global[None, :]) * HW
                           + safe_h[:, None] * W
                           + safe_w[:, None])
                x_load_mask = (k_mask[None, :] & m_mask[:, None]
                               & h_ok[:, None] & w_ok[:, None])
                x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0).to(tl.float16)

                w_addrs = (w_cl_ptr
                           + offs_n[None, :] * W_CO_STRIDE
                           + kh * W_KH_STRIDE
                           + kw * W_CIN_STRIDE
                           + cin_global[:, None])
                w_load_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0).to(tl.float16)

                acc += tl.dot(x_tile, w_tile)

        k_start += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + out_h[:, None] * W
                 + out_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Stage-2: Per-Tile 稀疏 1×1 Conv（按 tile 形状拆分 autotune）
# ═══════════════════════════════════════════════════════════════════════

@autotune(configs=_CONV1X1_CONFIGS_8x8,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def sparse_conv1x1_pertile_kernel_8x8(
    x_ptr, w_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    out_h = gh_idx * BLOCK_H + tile_bh
    out_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (out_h < H) & (out_w < W)
    safe_h = tl.minimum(out_h, H - 1)
    safe_w = tl.minimum(out_w, W - 1)

    HW: tl.constexpr = H * W

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        x_addrs = (x_ptr
                   + (n_idx * C_IN + cin_global[None, :]) * HW
                   + safe_h[:, None] * W
                   + safe_w[:, None])
        x_load_mask = k_mask[None, :] & m_mask[:, None]
        x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0).to(tl.float16)

        w_addrs = (w_ptr
                   + offs_n[None, :] * C_IN
                   + cin_global[:, None])
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0).to(tl.float16)

        acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + out_h[:, None] * W
                 + out_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


@autotune(configs=_CONV1X1_CONFIGS_8x16,
          key=['C_IN', 'C_OUT', 'H', 'W', 'GH', 'GW'])
@triton.jit
def sparse_conv1x1_pertile_kernel_8x16(
    x_ptr, w_ptr, bias_ptr, tile_ptr_data, tile_cin_ptr, y_ptr, N_val,
    C_IN: tl.constexpr, C_OUT: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    GH: tl.constexpr, GW: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # same body as 8x8 variant
    tile_id = tl.program_id(0)
    pid_cout = tl.program_id(1)

    total_tiles = N_val * GH * GW
    if tile_id >= total_tiles:
        return

    gw_idx = tile_id % GW
    tmp = tile_id // GW
    gh_idx = tmp % GH
    n_idx = tmp // GH

    c_out_start = pid_cout * BLOCK_N
    offs_n = c_out_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < C_OUT

    offs_m = tl.arange(0, BLOCK_M)
    tile_bh = offs_m // BLOCK_W
    tile_bw = offs_m % BLOCK_W
    out_h = gh_idx * BLOCK_H + tile_bh
    out_w = gw_idx * BLOCK_W + tile_bw
    m_mask = (out_h < H) & (out_w < W)
    safe_h = tl.minimum(out_h, H - 1)
    safe_w = tl.minimum(out_w, W - 1)

    HW: tl.constexpr = H * W

    tile_start = tl.load(tile_ptr_data + tile_id)
    tile_end = tl.load(tile_ptr_data + tile_id + 1)
    active_K = tile_end - tile_start

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_start = 0
    while k_start < active_K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < active_K

        cin_global = tl.load(
            tile_cin_ptr + tile_start + offs_k,
            mask=k_mask, other=0)

        x_addrs = (x_ptr
                   + (n_idx * C_IN + cin_global[None, :]) * HW
                   + safe_h[:, None] * W
                   + safe_w[:, None])
        x_load_mask = k_mask[None, :] & m_mask[:, None]
        x_tile = tl.load(x_addrs, mask=x_load_mask, other=0.0).to(tl.float16)

        w_addrs = (w_ptr
                   + offs_n[None, :] * C_IN
                   + cin_global[:, None])
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_addrs, mask=w_load_mask, other=0.0).to(tl.float16)

        acc += tl.dot(x_tile, w_tile)
        k_start += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias_vals[None, :]

    out_addrs = (y_ptr
                 + (n_idx * C_OUT + offs_n[None, :]) * HW
                 + out_h[:, None] * W
                 + out_w[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_addrs, acc, mask=out_mask)


# ═══════════════════════════════════════════════════════════════════════
# Legacy kernels — 向后兼容
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def prescan_kernel(
    x_ptr, flags_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    gw = pid % GRID_W; tmp = pid // GRID_W
    gh = tmp % GRID_H; tmp2 = tmp // GRID_H
    c = tmp2 % C; n = tmp2 // C
    hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask = (hh < H) & (ww < W)
    base = (n * C + c) * H
    val = tl.load(x_ptr + (base + hh) * W + ww, mask=mask, other=0.0)
    is_nz = tl.max(tl.abs(val)) > THRESHOLD
    tl.store(flags_ptr + pid, is_nz.to(tl.int32))


@triton.jit
def dense_conv3x3_kernel(
    x_ptr, y_ptr, N, C, H, W, GRID_H, GRID_W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * GRID_H * GRID_W
    if pid >= total:
        return
    gw = pid % GRID_W; tmp = pid // GRID_W
    gh = tmp % GRID_H; tmp2 = tmp // GRID_H
    c = tmp2 % C; n = tmp2 // C
    hh = (gh * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ww = (gw * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]
    mask_hw = (hh < H) & (ww < W)
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_idx = hh + kh; w_idx = ww + kw
            m = mask_hw & (h_idx >= 0) & (h_idx < H) & (w_idx >= 0) & (w_idx < W)
            acc += tl.load(x_ptr + ((n*C+c)*H+h_idx)*W+w_idx, mask=m, other=0.0)
    tl.store(y_ptr + ((n*C+c)*H+hh)*W+ww, acc, mask=mask_hw)


# ═══════════════════════════════════════════════════════════════════════
# 主入口: sparse_conv2d_forward
# ═══════════════════════════════════════════════════════════════════════

def sparse_conv2d_forward(x, weight, bias, block_size=None,
                          kernel_size=3, threshold=1e-6,
                          w_cl=None, counts_buf=None, tile_cin_buf=None,
                          return_ms=False):
    """
    Per-Tile 通道稀疏 Conv2d — autotune + 零 Host-Device 同步版。
    """
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    device = x.device

    # ── Tile 尺寸 ──
    BH, BW = _select_tile_sizes(H, W)
    GH = triton.cdiv(H, BH)
    GW = triton.cdiv(W, BW)
    N_TILES = N * GH * GW

    x_f16 = x.half().contiguous()

    # ── 权重 Channel-Last 转换 ──
    if w_cl is not None:
        w_cl_f16 = w_cl
    else:
        if kernel_size == 3:
            w_cl_f16 = weight.half().permute(0, 2, 3, 1).contiguous()
        else:
            w_cl_f16 = weight.half().reshape(C_OUT, C_IN).contiguous()

    # ── Buffer 准备 ──
    if counts_buf is None or counts_buf.numel() < N_TILES:
        counts_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
    if tile_cin_buf is None or tile_cin_buf.numel() < N_TILES * C_IN:
        tile_cin_buf = torch.empty(N_TILES * C_IN, dtype=torch.int32, device=device)

    # ── Stage-1: Per-Tile CSR（零同步） ──
    tile_ptr = _build_tile_csr(
        x_f16, N, C_IN, H, W, BH, BW, GH, GW,
        kernel_size, threshold,
        counts_buf=counts_buf, tile_cin_buf=tile_cin_buf)

    # ── Stage-2: Autotuned 稀疏 Conv ──
    has_bias = bias is not None
    bias_f32 = (bias.float().contiguous() if has_bias
                else torch.empty(1, device=device))
    y = torch.zeros(N, C_OUT, H, W, dtype=torch.float32, device=device)

    sparse_ms = 0.0
    if return_ms:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    def _grid(META):
        return (N_TILES, triton.cdiv(C_OUT, META['BLOCK_N']))

    # ── Stage-2 kernel dispatch: enforce Prescan BH/BW == Kernel BLOCK_H/W ──
    if BW == 16:
        _kernel_3x3 = sparse_conv3x3_pertile_kernel_8x16
        _kernel_1x1 = sparse_conv1x1_pertile_kernel_8x16
    else:
        _kernel_3x3 = sparse_conv3x3_pertile_kernel_8x8
        _kernel_1x1 = sparse_conv1x1_pertile_kernel_8x8

    if kernel_size == 3:
        (_kernel_3x3)[_grid](
            x_f16, w_cl_f16, bias_f32,
            tile_ptr, tile_cin_buf,
            y,
            N,
            C_IN, C_OUT, H, W,
            GH, GW,
            HAS_BIAS=has_bias,
        )
    else:
        (_kernel_1x1)[_grid](
            x_f16, w_cl_f16, bias_f32,
            tile_ptr, tile_cin_buf,
            y,
            N,
            C_IN, C_OUT, H, W,
            GH, GW,
            HAS_BIAS=has_bias,
        )

    if return_ms:
        end_evt.record()
        torch.cuda.synchronize(device)
        sparse_ms = start_evt.elapsed_time(end_evt)

    return y, sparse_ms