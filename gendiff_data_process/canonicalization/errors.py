from __future__ import annotations

from typing import Any


class CanonicalizationError(ValueError):
    """带稳定错误码的失败关闭异常。"""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.context = context
