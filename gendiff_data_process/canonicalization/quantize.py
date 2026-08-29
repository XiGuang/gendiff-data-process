from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Iterable

from .errors import CanonicalizationError
from .types import PointQ


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise CanonicalizationError("E_NONFINITE_VALUE", "布尔值不是合法坐标")
    try:
        if isinstance(value, float) and not math.isfinite(value):
            raise CanonicalizationError("E_NONFINITE_VALUE", "坐标包含 NaN/Inf", value=value)
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise CanonicalizationError("E_NONFINITE_VALUE", "坐标不是有限数值", value=value) from exc
    if not result.is_finite():
        raise CanonicalizationError("E_NONFINITE_VALUE", "坐标包含 NaN/Inf", value=str(value))
    return result


def quantize_scalar(value: int | float | str | Decimal, grid: str | Decimal) -> int:
    coordinate = _decimal(value)
    step = _decimal(grid)
    if step <= 0:
        raise ValueError("量化网格必须为正数")
    magnitude = (abs(coordinate) / step + Decimal("0.5")).to_integral_value(rounding=ROUND_FLOOR)
    return int(magnitude.copy_negate() if coordinate < 0 else magnitude)


def dequantize_scalar(value_q: int, grid: str | Decimal) -> Decimal:
    return Decimal(value_q) * _decimal(grid)


def quantize_points(points: Iterable[tuple[object, object]], grid: str | Decimal) -> tuple[PointQ, ...]:
    return quantize_points_with_collapse(points, grid)[0]


def quantize_points_with_collapse(
    points: Iterable[tuple[object, object]],
    grid: str | Decimal,
) -> tuple[tuple[PointQ, ...], bool]:
    normalized = tuple((_decimal(x), _decimal(z)) for x, z in points)
    quantized = tuple((quantize_scalar(x, grid), quantize_scalar(z, grid)) for x, z in normalized)
    collapsed = len(set(quantized)) < len(set(normalized))
    return quantized, collapsed
