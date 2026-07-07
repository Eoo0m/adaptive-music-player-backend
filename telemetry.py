import contextvars
import time
import uuid


request_id_var = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def get_request_id() -> str:
    return request_id_var.get()


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> float:
    return now_ms() - start_ms
