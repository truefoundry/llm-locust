import dataclasses
import math
import time


@dataclasses.dataclass
class TriggerShutdown:
    """used for triggering shutdown of a proces or thread"""


@dataclasses.dataclass
class SetStartTime:
    """Tracks start_time of UserSpawner loop"""

    start_time: int


@dataclasses.dataclass
class SetLastProcessedTime:
    """Tracks current time of UserSpawner loop"""

    current_time: int


@dataclasses.dataclass
class SetActiveUsers:
    """Tracks changes in active user count"""

    total_users: int


@dataclasses.dataclass
class SetUserInfo:
    """Sets changes in user count in User Loop"""

    max_users: int
    user_addition_count: int
    user_addition_time: float


@dataclasses.dataclass
class RequestFailureLog:
    """Records information about failed requests"""

    timestamp: int
    start_time: float
    end_time: float
    status_code: int


@dataclasses.dataclass
class RequestSuccessLog:
    """Records information about successful requests"""

    result_chunks: list[bytes]
    num_input_tokens: int
    timestamp: int
    token_times: list[float]
    start_time: float
    end_time: float
    status_code: int


@dataclasses.dataclass
class MetricsLog:
    """Records metrics based on time and active user"""

    timestamp: int
    data: int | float


@dataclasses.dataclass
class ErrorLog:
    """Records error information"""

    error_message: str


def get_timestamp_seconds() -> int:
    """Get current time floor value"""
    return math.floor(time.time())
