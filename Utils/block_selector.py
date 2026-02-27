"""
收益评估与动态分块 — 根据特征图空间尺寸选择最优 block 大小

策略（基于 benchmark 数据校准）：
  - H >= 32  → Block = 16  (大型特征图，Triton kernel 效率最高)
  - 16 <= H < 32 → Block = 8   (中型特征图)
  - 8 <= H < 16  → Block = 4   (小型特征图，稀疏优化的下界)
  - H < 8         → None        (微型特征图，Gating 开销过大，保持原生稠密算子)
"""

from typing import Optional


def select_block_size(H: int, W: int) -> Optional[int]:
    """
    根据特征图高度和宽度自动选择 block 大小。

    收益评估逻辑：
      - block_size 越大，每次 prescan 检查的区域越大，但大 block 中出现非零的概率也更高
      - block_size 越小，prescan 粒度更细，跳零更精确，但 kernel launch 开销相对更大
      - H < 8 时，整个特征图只有几个 block，prescan + gather 的 overhead 超过跳零收益

    Args:
        H: 特征图高度
        W: 特征图宽度

    Returns:
        block_size: int or None (None 表示不适合稀疏加速，应保持原生算子)
    """
    spatial = min(H, W)

    if spatial >= 32:
        return 16
    elif spatial >= 16:
        return 8
    elif spatial >= 8:
        return 4
    else:
        # 微型特征图 (e.g. 7×7)，Gating 开销 > 跳零收益
        return None