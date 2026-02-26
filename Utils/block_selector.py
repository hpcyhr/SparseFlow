"""
自动 block 大小选择策略

根据特征图空间尺寸选择最优 block 大小：
  - H/W >= 56  -> Block = 16  (layer1: 56x56, layer2: 28x28)
  - 14 <= H/W < 56  -> Block = 8   (layer3: 14x14)
  - H/W <= 7   -> None (跳过，不使用稀疏加速)
"""

from typing import Optional


def select_block_size(H: int, W: int) -> Optional[int]:
    """
    根据特征图高度和宽度自动选择 block 大小。

    Args:
        H: 特征图高度
        W: 特征图宽度

    Returns:
        block_size: int or None (None 表示不适合稀疏加速)
    """
    spatial = min(H, W)
    if spatial >= 56:
        return 16
    elif spatial >= 14:
        return 8
    else:
        return None