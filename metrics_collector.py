import logging
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread

from clients import BaseModelClient
from metrics import LLMMetricsList
from utils import (
    ErrorLog,
    RequestFailureLog,
    RequestSuccessLog,
    SetActiveUsers,
    SetLastProcessedTime,
    SetStartTime,
    TriggerShutdown,
    get_timestamp_seconds,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Handles collection of metrics via metrics queue, and their aggregation and logging"""

    def __init__(
        self,
        global_state: object,
        metrics_queue: Queue,
        model_client: BaseModelClient,
        metrics_window_size: int = 30,
        quantiles: list[int] = None,
    ):
        self.global_state = global_state
        self.start_time = 0
        self.metrics_list = LLMMetricsList(quantiles)
        self.on_going_users: int = 0
        self.quantiles: list[int] = quantiles if quantiles is not None else [50]
        self.metrics_window_size: int = metrics_window_size
        self.computed_metrics_task = None
        self.collection_task = None
        self.metrics_queue = metrics_queue
        self.model_client = model_client
        self.last_processed_request_time: int = 0
        self.running = False

    def logging_function(self, log_dict: dict) -> None:
        """writes log entries to the JSONL file and ML Repo"""
        try:
            self.global_state.update(log_dict)
        except Exception as e:
            print(e)

    def start_logging(self) -> None:
        """Start worker threads"""
        self.running = True
        self.collection_task = Thread(target=self.collect_metrics)
        self.collection_task.start()
        self.computed_metrics_task = Thread(
            target=self.report_metrics,
            kwargs={
                "sliding_window_size": self.metrics_window_size,
                "sliding_window_stride": 2,
                "metric_function": self.log_metrics,
            },
        )
        self.computed_metrics_task.start()

    def stop_logging(self) -> None:
        self.running = False
        self.collection_task.join()
        self.computed_metrics_task.join()

    def collect_metrics(self) -> None:
        """Collects all metrics via the metrics queue"""
        while self.running:
            try:
                metrics_data = self.metrics_queue.get(timeout=1)
                if isinstance(metrics_data, TriggerShutdown):
                    return
                if isinstance(metrics_data, RequestSuccessLog | RequestFailureLog):
                    self.metrics_list.collect_request(metrics_data, self)
                elif isinstance(metrics_data, ErrorLog):
                    logger.warning(f"Error in User Loop: {metrics_data.error_message}")
                elif isinstance(metrics_data, SetActiveUsers):
                    self.on_going_users = metrics_data.total_users
                elif isinstance(metrics_data, SetStartTime):
                    self.start_time = metrics_data.start_time
                elif isinstance(metrics_data, SetLastProcessedTime):
                    self.last_processed_request_time = metrics_data.current_time
            except Empty:
                continue

    def log_metrics(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> None:
        """
        Log metrics using logging function. Filter the metrics from start_timestamp --> end_timestamp
        """
        if self.last_processed_request_time > 0:
            # processing is kept ahead to ensure metrics are collected fully
            while self.running and self.last_processed_request_time < end_timestamp:
                time.sleep(max(0, end_timestamp - self.last_processed_request_time))

        self.logging_function(
            {
                "active_users": self.on_going_users,
                **self.metrics_list.calculate(start_timestamp, end_timestamp),
            }
        )

    def report_metrics(
        self,
        sliding_window_size: int,
        sliding_window_stride: int,
        metric_function: callable,
    ) -> None:
        """
        This function will report the metrics using metric_function in the past sliding_window_size seconds, update every sliding_window_stride seconds
        """
        while self.start_time == 0:
            time.sleep(0.1)  # wait for user spawner to start

        while self.running:
            # Log the metrics
            end_timestamp = get_timestamp_seconds()
            metric_function(
                start_timestamp=max(
                    end_timestamp - sliding_window_size, self.start_time
                ),
                end_timestamp=end_timestamp,
            )
            time.sleep(sliding_window_stride)
