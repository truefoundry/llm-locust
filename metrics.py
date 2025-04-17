from abc import ABC, abstractmethod

import numpy as np

from utils import MetricsLog, RequestFailureLog, RequestSuccessLog


def filter_bucket(
    bucket: list[MetricsLog],
    start_timestamp: int,
    end_timestamp: int,
) -> list[int | float]:
    """
    Filter a bucket from start_timestamp --> end_timestamp
    """
    return [
        m.data
        for m in list(bucket)
        if (m.timestamp >= start_timestamp and m.timestamp < end_timestamp)
    ]


class BaseTimeMetric:
    """Base class for storing metrics in a bucket"""

    def __init__(self) -> None:
        self.bucket: list[MetricsLog] = []

    def collect(self, data: MetricsLog) -> None:
        """Collect the metric data into a bucket"""
        self.bucket.append(data)

    def reset(self) -> None:
        """Empty the bucket"""
        self.bucket = []

    def _get_filtered_bucket(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> list[int | float]:
        """Helper method to get filtered bucket data"""
        return filter_bucket(self.bucket, start_timestamp, end_timestamp)


class SimpleMetric(BaseTimeMetric, ABC):
    """Metric class for simple calculations"""

    @abstractmethod
    def calculate(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> dict[str, int | float]:
        """Calculate the metrics and return a dictionary with display name as key and metric value as value"""
        raise NotImplementedError

    @abstractmethod
    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        """Parse metric from request log and collect in bucket"""
        raise NotImplementedError


class QuantileMetric(SimpleMetric, ABC):
    """Metric class for quantile-based calculations"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        raise NotImplementedError

    def __init__(self, quantiles: list[int]) -> None:
        self.quantiles = quantiles
        super().__init__()

    def calculate(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> dict[str, int | float]:
        """Calculate quantiles on the bucket"""
        bucket = self._get_filtered_bucket(start_timestamp, end_timestamp)
        log_dict = {f"{self.name}_quantile_{q}": 0 for q in self.quantiles}
        if bucket:
            log_dict.update(
                {
                    f"{self.name}_quantile_{q}": np.percentile(bucket, q)
                    for q in self.quantiles
                }
            )
        return log_dict


class ResponseMetric(SimpleMetric):
    """Metrics based on basic requests info"""

    def calculate(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> dict[str, int | float]:
        bucket = self._get_filtered_bucket(start_timestamp, end_timestamp)
        time_interval = end_timestamp - start_timestamp
        if time_interval > 0:
            return {
                "failed_requests_per_second": sum(
                    1 for status in bucket if status != 200
                )
                / time_interval,
                "requests_per_second": sum(1 for status in bucket if status == 200)
                / time_interval,
            }
        return {
            "failed_requests_per_second": 0,
            "requests_per_second": 0,
        }

    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        self.collect(
            MetricsLog(
                timestamp=request_log.timestamp,
                data=request_log.status_code,
            )
        )


class OutputTokensMetric(SimpleMetric):
    """Metrics based on output tokens"""

    @property
    def name(self) -> str:
        return "output_tokens_per_request"

    def calculate(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> dict[str, int | float]:
        bucket = self._get_filtered_bucket(start_timestamp, end_timestamp)
        time_interval = end_timestamp - start_timestamp
        log_dict = {"total_output_tokens_per_second": 0}
        if time_interval > 0:
            log_dict.update(
                {
                    "total_output_tokens_per_second": sum(bucket) / time_interval,
                }
            )
        return log_dict

    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        if isinstance(request_log, RequestFailureLog):
            return
        result = []
        for chunk in request_log.result_chunks:
            output = metrics_collector.model_client.parse_response(chunk)
            result += output
        self.collect(
            MetricsLog(
                timestamp=request_log.timestamp,
                data=len(result),
            )
        )


class EmptyTokensMetric(SimpleMetric):
    """Metrics based on output tokens"""

    def calculate(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> dict[str, int | float]:
        bucket = self._get_filtered_bucket(start_timestamp, end_timestamp)
        time_interval = end_timestamp - start_timestamp
        log_dict = {"total_empty_output_tokens_per_second": 0}
        if time_interval > 0:
            log_dict.update(
                {
                    "total_empty_output_tokens_per_second": sum(bucket) / time_interval,
                }
            )
        return log_dict

    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        if isinstance(request_log, RequestFailureLog):
            return
        for chunk in request_log.result_chunks:
            output = metrics_collector.model_client.parse_response(chunk)
            if len(output) == 0:
                self.collect(
                    MetricsLog(
                        timestamp=request_log.timestamp,
                        data=1,
                    )
                )


class TTFTMetric(QuantileMetric):
    @property
    def name(self) -> str:
        return "response_time_first_token_ms"

    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        if isinstance(request_log, RequestFailureLog):
            return
        for token_time, chunk in zip(
            request_log.token_times,
            request_log.result_chunks,
            strict=True,
        ):
            output = metrics_collector.model_client.parse_response(chunk)
            if output:
                ttft = token_time - request_log.start_time
                self.collect(
                    MetricsLog(
                        timestamp=request_log.timestamp,
                        data=1000 * ttft,
                    )
                )
                break


class InterTokenLatencyMetric(QuantileMetric):
    @property
    def name(self) -> str:
        return "inter_token_latency_ms"

    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        if isinstance(request_log, RequestFailureLog):
            return
        prev_time = None
        for token_time, chunk in zip(
            request_log.token_times,
            request_log.result_chunks,
            strict=True,
        ):
            output = metrics_collector.model_client.parse_response(chunk)
            if output:
                if prev_time:
                    self.collect(
                        MetricsLog(
                            timestamp=request_log.timestamp,
                            data=1000 * ((token_time - prev_time) / len(output)),
                        )
                    )
                prev_time = token_time


class ResponseLatencyMetric(QuantileMetric):
    @property
    def name(self) -> str:
        return "response_time_seconds"

    def collect_request(
        self,
        request_log: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object | None = None,
    ) -> None:
        if isinstance(request_log, RequestFailureLog):
            return
        self.collect(
            MetricsLog(
                timestamp=request_log.timestamp,
                data=request_log.end_time - request_log.start_time,
            )
        )


class MetricsList:
    def __init__(self) -> None:
        self.metrics: list[SimpleMetric] = []

    def collect_request(
        self,
        metrics_data: RequestSuccessLog | RequestFailureLog,
        metrics_collector: object,
    ) -> None:
        for metric in self.metrics:
            metric.collect_request(
                request_log=metrics_data, metrics_collector=metrics_collector
            )

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def calculate(
        self, start_timestamp: int, end_timestamp: int
    ) -> dict[str, int | float]:
        stats = {}
        for metric in self.metrics:
            stats.update(
                metric.calculate(
                    start_timestamp=start_timestamp, end_timestamp=end_timestamp
                )
            )
        return stats


class MinimalMetricsList(MetricsList):
    def __init__(self, quantiles: list[int]) -> None:
        super().__init__()
        self.metrics.extend(
            [
                ResponseMetric(),
                ResponseLatencyMetric(quantiles),
            ]
        )


class LLMMetricsList(MinimalMetricsList):
    def __init__(self, quantiles: list[int]) -> None:
        super().__init__(quantiles=quantiles)
        self.metrics.extend(
            [
                EmptyTokensMetric(),
                OutputTokensMetric(),
                TTFTMetric(quantiles),
                InterTokenLatencyMetric(quantiles),
            ]
        )
