from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


def canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {unicodedata.normalize("NFC", str(key)): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON 禁止 NaN/Inf")
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    raise TypeError(f"不支持的 canonical JSON 类型: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
