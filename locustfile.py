"""
Locust load test for LLM chat completions endpoint (streaming only).
SSE streaming; one stats entry per request for correct RPS.
"""

import os
import random
import time
import logging
from locust import task
from locust.contrib.fasthttp import FastHttpUser

# Setup simple logging for request debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("locustfile")

# ---------------------------------------------------------------------------
# Prompt pool to avoid server-side caching bias
# ---------------------------------------------------------------------------

PROMPT_POOL = [
    "Explain quantum entanglement in one paragraph.",
    "Write a haiku about distributed systems.",
    "What are the trade-offs between consistency and availability in distributed databases?",
    "List three best practices for microservice observability.",
    "Describe the CAP theorem briefly.",
    "How would you debug a memory leak in a long-running Python process?",
    "Summarize the benefits of event-driven architecture.",
    "What is idempotency and why does it matter in APIs?",
    "Give a short definition of eventual consistency.",
    "Name two strategies for handling backpressure in streaming systems.",
    "What distinguishes a circuit breaker from a retry with exponential backoff?",
    "Explain the difference between synchronous and asynchronous replication.",
    "Write two sentences about the fallacies of distributed computing.",
    "What is a dead letter queue used for?",
    "Describe one approach to achieve exactly-once processing in stream processing.",
]


def get_random_messages() -> list[dict[str, str]]:
    """Return a random prompt as OpenAI-style messages."""
    prompt = random.choice(PROMPT_POOL)
    return [{"role": "user", "content": prompt}]


def get_auth_headers() -> dict[str, str]:
    """Bearer token from environment."""
    logger.info(f"Getting auth headers with token: {os.environ.get('LOCUST_AUTH_TOKEN', 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImlKQm4tTW04c1BMdHRhbzlLSEFxeDJpUjIyayJ9.eyJhdWQiOiI3NjY1NjQ2MS02ZTc0LTYxN2EtM2E2Ni0zODM3MzQ2MjM4MzkiLCJleHAiOjM3MzI2MDI0NjIsImlhdCI6MTc3MzA1MDQ2MiwiaXNzIjoidHJ1ZWZvdW5kcnkuY29tIiwic3ViIjoiY21tajBncm15MDAwazAxbXU2OWdpYXdtMSIsImp0aSI6ImNtbWowZ3JvaDAwMGwwMW11M3R0MDRzdmwiLCJzdWJqZWN0U2x1ZyI6ImRlZmF1bHQtY21tajA3NjRtMDAwMzAxbnpmY2o2ZmU2ayIsInVzZXJuYW1lIjoiZGVmYXVsdC1jbW1qMDc2NG0wMDAzMDFuemZjajZmZTZrIiwidXNlclR5cGUiOiJzZXJ2aWNlYWNjb3VudCIsInN1YmplY3RUeXBlIjoic2VydmljZWFjY291bnQiLCJ0ZW5hbnROYW1lIjoidmVkYW50YXoiLCJyb2xlcyI6W10sImp3dElkIjoiY21tajBncm9oMDAwbDAxbXUzdHQwNHN2bCIsImFwcGxpY2F0aW9uSWQiOiI3NjY1NjQ2MS02ZTc0LTYxN2EtM2E2Ni0zODM3MzQ2MjM4MzkifQ.eZl3CgV6NUH2SqExkILGaHOI1DCn6aspnMLJZn2mbenw4bjuonYhKsloL9gsp_8FDNZXYINJeb9epPBDxBcKmevGxmDnSS40fRMdQn3NKWOXYGYISmmpmtQG7LPhaMIdlqYIahrkl1_5Zq8b3pTN-u01gEP-lTOfOZDsMtvx1ExzmKYYelCdRo9cPPLJWKrNeRzBYFWaFP-Hx6bqON4D9HR9DvkjiKI0FQDNaunKh4yVxpqO66Y_CltEHSgSGwkF80oWutH-SZtwJgF0QYxYHGOtrofCpjWbYhqnbOa9FTsG-E31WlQRyymT4WkfUbYI1veF78D2MAejbJ6kcGeJHg')}")
    token = os.environ.get("LOCUST_AUTH_TOKEN", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImlKQm4tTW04c1BMdHRhbzlLSEFxeDJpUjIyayJ9.eyJhdWQiOiI3NjY1NjQ2MS02ZTc0LTYxN2EtM2E2Ni0zODM3MzQ2MjM4MzkiLCJleHAiOjM3MzI2MDI0NjIsImlhdCI6MTc3MzA1MDQ2MiwiaXNzIjoidHJ1ZWZvdW5kcnkuY29tIiwic3ViIjoiY21tajBncm15MDAwazAxbXU2OWdpYXdtMSIsImp0aSI6ImNtbWowZ3JvaDAwMGwwMW11M3R0MDRzdmwiLCJzdWJqZWN0U2x1ZyI6ImRlZmF1bHQtY21tajA3NjRtMDAwMzAxbnpmY2o2ZmU2ayIsInVzZXJuYW1lIjoiZGVmYXVsdC1jbW1qMDc2NG0wMDAzMDFuemZjajZmZTZrIiwidXNlclR5cGUiOiJzZXJ2aWNlYWNjb3VudCIsInN1YmplY3RUeXBlIjoic2VydmljZWFjY291bnQiLCJ0ZW5hbnROYW1lIjoidmVkYW50YXoiLCJyb2xlcyI6W10sImp3dElkIjoiY21tajBncm9oMDAwbDAxbXUzdHQwNHN2bCIsImFwcGxpY2F0aW9uSWQiOiI3NjY1NjQ2MS02ZTc0LTYxN2EtM2E2Ni0zODM3MzQ2MjM4MzkifQ.eZl3CgV6NUH2SqExkILGaHOI1DCn6aspnMLJZn2mbenw4bjuonYhKsloL9gsp_8FDNZXYINJeb9epPBDxBcKmevGxmDnSS40fRMdQn3NKWOXYGYISmmpmtQG7LPhaMIdlqYIahrkl1_5Zq8b3pTN-u01gEP-lTOfOZDsMtvx1ExzmKYYelCdRo9cPPLJWKrNeRzBYFWaFP-Hx6bqON4D9HR9DvkjiKI0FQDNaunKh4yVxpqO66Y_CltEHSgSGwkF80oWutH-SZtwJgF0QYxYHGOtrofCpjWbYhqnbOa9FTsG-E31WlQRyymT4WkfUbYI1veF78D2MAejbJ6kcGeJHg")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# LLM Streaming User — SSE responses, full stream duration as response_time
# ---------------------------------------------------------------------------

def _drain_stream(response, start: float) -> tuple[float | None, int]:
    """
    Drain the response body without reading/parsing content.
    Returns (ttft_ms, chunk_count). TTFT = time to first chunk; chunk_count for response_length.
    """
    ttft_ms: float | None = None
    chunk_count = 0

    for chunk in response.iter_content(chunk_size=1024, decode_content=False):
        if chunk is None:
            continue
        chunk_count += 1
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - start) * 1000
    return ttft_ms, chunk_count


class LLMStreamingUser(FastHttpUser):
    """Streaming chat completions; measures TTFT and full stream duration."""

    tags = ["streaming"]

    @task(10)
    def chat_stream(self):
        path = "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **get_auth_headers(),
        }
        body = {
            "model": "load-test-models/model1",
            "messages": get_random_messages(),
            "stream": True,
        }

        # Logging request details
        logger.info(
            f"Sending POST {path} | Headers: {headers} | Body: {body}"
        )

        start = time.perf_counter()

        with self.client.post(
            path,
            headers=headers,
            json=body,
            stream=True,
            catch_response=True,
            name="chat_completions_stream",
        ) as response:
            logger.info(f"Response: {response}")
            logger.info(f"Received response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Request failed with status code: {response.status_code}")
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                _, chunk_count = _drain_stream(response, start)
                logger.info(f"Streamed {chunk_count} chunks in response.")
            except Exception as e:
                total_ms = (time.perf_counter() - start) * 1000
                logger.error(f"Error during stream drain: {e}")
                response.request_meta["response_time"] = total_ms
                response.request_meta["response_length"] = 0
                response.failure(str(e))
                return

            total_ms = (time.perf_counter() - start) * 1000
            # Report full stream duration so the main request time reflects end-to-end streaming
            response.request_meta["response_time"] = total_ms
            response.request_meta["response_length"] = chunk_count
            logger.info(
                f"Request complete in {total_ms:.2f}ms | Chunks: {chunk_count}"
            )
            response.success()
