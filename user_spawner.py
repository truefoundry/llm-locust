import asyncio
import logging
from asyncio.tasks import Task
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Queue
from queue import Empty

from clients import BaseModelClient
from user import User
from utils import (
    SetActiveUsers,
    SetLastProcessedTime,
    SetStartTime,
    SetUserInfo,
    TriggerShutdown,
    get_timestamp_seconds,
)

logger = logging.getLogger(__name__)


class UserSpawner:
    """Handles spawning of users and each users requests"""

    def __init__(
        self,
        model_client: BaseModelClient,
        metrics_queue: Queue,
        max_user_count: int = 1,
        user_addition_count: int = 1,
        user_addition_time: float = 0,
    ):
        self.metrics_queue = metrics_queue
        self.model_client = model_client
        self.max_user_count = max_user_count
        self.user_addition_count = user_addition_count
        self.user_addition_time = user_addition_time
        self.current_user_id = -1
        self.user_list: list[Task] = []
        self.shutdown = False
        self.user_updated_time = get_timestamp_seconds()

    def spawn_user(self) -> None:
        """Spawn a new user"""
        self.current_user_id = self.current_user_id + 1
        user = User(self.model_client, self.metrics_queue, self.current_user_id)
        self.user_list.append(user)

    async def cancel_all_users(self) -> None:
        """Cancel all users gracefully"""
        self.shutdown = True
        logger.info("stopping users")
        await asyncio.gather(*(user.stop() for user in self.user_list))
        logger.info("stopped users")
        self.user_list = []

    async def spawner_loop(self) -> None:
        """
        Handles dynamic addition and removal of users based on current parameters.
        Parameters can change at any time through user_control_queue.
        """
        start_time = get_timestamp_seconds()
        self.metrics_queue.put(SetStartTime(start_time=start_time))

        while not self.shutdown:
            current_time = get_timestamp_seconds()
            self.metrics_queue.put(SetLastProcessedTime(current_time=current_time))
            target_users = self.max_user_count
            # Calculate target user count for current time
            if self.user_addition_time > 0:
                elapsed_intervals = 1 + (
                    (current_time - self.user_updated_time) // self.user_addition_time
                )
                interval_change = elapsed_intervals * self.user_addition_count
                target, operator = (
                    (len(self.user_list) - interval_change, max)
                    if len(self.user_list) > self.max_user_count
                    else (interval_change, min)
                )
                target_users = operator(
                    target,  # Ramp up/down
                    self.max_user_count,  # Don't go above/below new max
                )
            current_users = len(self.user_list)
            # Handle user addition
            if current_users < target_users:
                users_to_add = min(
                    self.user_addition_count,  # Don't add more than specified batch size
                    target_users - current_users,  # Don't exceed target
                )
                for _ in range(users_to_add):
                    if len(self.user_list) >= self.max_user_count:
                        break
                    self.spawn_user()

            # Handle user removal if max_users was decreased
            elif current_users > target_users:
                num_users_to_remove = min(
                    self.user_addition_count, current_users - target_users
                )
                users_to_remove = []
                for _ in range(num_users_to_remove):
                    if self.user_list:
                        user = self.user_list.pop()
                        users_to_remove.append(user)
                await asyncio.gather(*(user.stop() for user in users_to_remove))

            self.metrics_queue.put(SetActiveUsers(total_users=len(self.user_list)))
            await asyncio.sleep(0.25)  # Shorter sleep for more responsive scaling


async def start_user_loop(
    max_users: int,
    user_addition_count: int,
    user_addition_time: float,
    model_client: BaseModelClient,
    metrics_queue: Queue,
    user_control_queue: Queue,
) -> int:
    """Starts User Spawner loop, runs till session time and stops"""
    user_spawner = UserSpawner(
        model_client=model_client,
        metrics_queue=metrics_queue,
        max_user_count=max_users,
        user_addition_count=user_addition_count,
        user_addition_time=user_addition_time,
    )
    # start user spanwer loop
    asyncio.create_task(user_spawner.spawner_loop())
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    while True:
        try:
            # Run queue.get in thread pool to avoid blocking event loop
            user_control_msg = await loop.run_in_executor(
                executor, partial(user_control_queue.get, timeout=10)
            )

            if isinstance(user_control_msg, TriggerShutdown):
                executor.shutdown(wait=False)
                await user_spawner.cancel_all_users()
                metrics_queue.put(TriggerShutdown())
                return

            if isinstance(user_control_msg, SetUserInfo):
                user_spawner.max_user_count = user_control_msg.max_users
                user_spawner.user_addition_count = user_control_msg.user_addition_count
                user_spawner.user_addition_time = user_control_msg.user_addition_time
                user_spawner.user_updated_time = get_timestamp_seconds()
        except Empty:
            continue
