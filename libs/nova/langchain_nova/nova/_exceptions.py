from typing import Any


class BadRequestError(Exception):
    def __init__(
        self,
        original_exception: Exception,
        request_state: Any = None,
        message: str = "",
    ) -> None:
        """ """
        self.original_exception = original_exception
        self.request_state = request_state if request_state is not None else {}
        self.message = message
        super().__init__(str(original_exception))
