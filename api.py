import argparse
import asyncio
import csv
import logging
import logging.config
import time
from io import StringIO
from multiprocessing import Process, Queue
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer

from clients import BaseModelClient, OpenAIChatStreamingClient
from metrics_collector import MetricsCollector
from prompt import (
    get_prompt_set,
    get_prompt_set_random,
    get_prompt_set_single,
    system_prompt,
)
from user_spawner import start_user_loop
from utils import SetUserInfo, TriggerShutdown


# Start the User Loop
def start_user_loop_async(
    model_client: BaseModelClient,
    metrics_queue: Queue,
    user_control_queue: Queue,
    user_addition_count: int,
    user_addition_time: float,
    max_users: int,
) -> None:
    logger.info("Starting User Spawner")
    asyncio.run(
        start_user_loop(
            max_users=max_users,
            user_addition_count=user_addition_count,
            user_addition_time=user_addition_time,
            model_client=model_client,
            metrics_queue=metrics_queue,
            user_control_queue=user_control_queue,
        )
    )


def str_to_bool(value: str) -> bool:
    """Convert string representation to boolean."""
    return value.lower() == "true"


def optional_str(value: str) -> str | None:
    """Convert 'none' string to None, otherwise return the string."""
    return None if value.lower() == "none" else value


def parse_quantiles(value: str) -> list[int]:
    """Parse comma-separated quantiles string into list of integers."""
    return sorted(int(x.strip()) for x in value.split(","))


def update_args(  # noqa: C901
    tokenizer: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    prompt_min_tokens: int | None = None,
    prompt_max_tokens: int | None = None,
    use_random_prompts: bool | None = None,
    use_single_prompt: bool | None = None,
    ignore_eos: bool | None = None,
    user_count: int | None = None,
    host: str | None = None,
    spawn_rate: float | None = None,
) -> None:
    """Update command line arguments with form values if provided."""
    if tokenizer:
        args.tokenizer = tokenizer
    if model:
        args.model = model
    if max_tokens:
        args.max_tokens = max_tokens
    if prompt_min_tokens:
        args.prompt_min_tokens = prompt_min_tokens
    if prompt_max_tokens:
        args.prompt_max_tokens = prompt_max_tokens
    if use_random_prompts is not None:
        args.use_random_prompts = use_random_prompts
    if use_single_prompt is not None:
        args.use_single_prompt = use_single_prompt
    if ignore_eos is not None:
        args.ignore_eos = ignore_eos
    if user_count is not None:
        args.user_count = user_count
    if host is not None:
        args.host = host
    if spawn_rate is not None:
        args.spawn_rate = spawn_rate


# arg parsing
parser = argparse.ArgumentParser(description="Benchmark LLM")
parser.add_argument(
    "--tokenizer", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct"
)
parser.add_argument(
    "--model", type=str, default="nousresearch-meta-llama-3-1-8b-instruct"
)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--metrics_logging_interval", type=int, default=30)
parser.add_argument("--prompt_min_tokens", type=int, default=10)
parser.add_argument("--prompt_max_tokens", type=int, default=1000)
parser.add_argument("--quantiles", type=parse_quantiles, default="50,90,99")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--use_random_prompts", type=str_to_bool, default="False")
parser.add_argument("--use_single_prompt", type=str_to_bool, default="False")
parser.add_argument("--ignore_eos", type=str_to_bool, default="False")
parser.add_argument("--user_count", type=int, default=1)
parser.add_argument("--host", type=str, default="http://localhost:8000")
parser.add_argument("--spawn_rate", type=float, default=1.0)

args = parser.parse_args()
app = FastAPI(title="LLM Load Testing API")
(
    STATE_INIT,
    STATE_SPAWNING,
    STATE_RUNNING,
    STATE_CLEANUP,
    STATE_STOPPING,
    STATE_STOPPED,
    STATE_MISSING,
) = [
    "ready",
    "spawning",
    "running",
    "cleanup",
    "stopping",
    "stopped",
    "missing",
]


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def create_dynamic_charts_config(
    log_data: dict[str, int | float],
) -> list[dict[str, str | list[dict[str, str]]]]:
    """Creates chart configurations from log data by grouping related metrics."""
    colors = {
        "basic": ["#4caf50", "#f44336"],
        "multi": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    }

    metric_groups = {
        "Requests per Second": [
            ("requests_per_second", "RPS"),
            ("failed_requests_per_second", "Failed RPS"),
        ],
        "Active Users": [("active_users", "Users")],
        "Tokens per Second": [
            ("total_output_tokens_per_second", "Output Tokens/s"),
            ("total_empty_output_tokens_per_second", "Empty Output Tokens/s"),
        ],
        "CPU & GPU Usage": [
            ("cpu_usage", "CPU Usage"),
            ("dcgm_gpu_utilization", "GPU Utilization"),
        ],
        "Memory Usage (MB)": [
            ("memory_usage", "Memory Usage"),
            ("dcgm_gpu_memory_used", "GPU Memory Used"),
        ],
    }

    charts = []

    # Process regular metrics
    for title, metrics in metric_groups.items():
        lines = []
        for key, name in metrics:
            matching_keys = [k for k in log_data if k.startswith(key)]
            for match_key in matching_keys:
                if match_key in log_data:
                    lines.append(
                        {
                            "name": name,
                            "key": snake_to_camel(match_key),
                            "value": log_data[match_key],
                        }
                    )

        if lines:
            charts.append(
                {
                    "title": title,
                    "lines": lines,
                    "colors": colors["multi"] if len(lines) > 2 else colors["basic"],
                }
            )

    # Process quantile metrics
    quantile_metrics = {}
    for key in log_data:
        if "_quantile_" in key:
            base, quantile = key.split("_quantile_")
            quantile_metrics.setdefault(base, []).append((key, quantile))

    for base, quantiles in quantile_metrics.items():
        if quantiles:
            lines = [
                {
                    "name": f"{int(quantile)}th percentile",
                    "key": snake_to_camel(key),
                    "value": log_data[key],
                }
                for key, quantile in sorted(quantiles)
            ]

            charts.append(
                {
                    "title": base.replace("_", " ").title(),
                    "lines": lines,
                    "colors": colors["multi"][: len(lines)],
                }
            )

    return charts


# Global state management
class GlobalState:
    """Stores global state of metrics"""

    def __init__(self) -> None:
        self.run_state = STATE_INIT
        self.charts = []
        self.history = []
        self.reset_counter = -1
        self.reset()

    def reset(self) -> None:
        self.start_time = None
        self.reset_counter += 1
        self.stats = [
            {
                "avg_content_length": 0,
                "avg_response_time": 0.0,
                "current_fail_per_sec": 0,
                "current_rps": 0,
                "max_response_time": 0.0,
                "median_response_time": 0,
                "method": None,
                "min_response_time": 0,
                "name": "Aggregated",
                "num_failures": 0,
                "num_requests": 0,
                "response_time_percentile_0.95": 0,
                "response_time_percentile_0.99": 0,
                "safe_name": "Aggregated",
                "total_fail_per_sec": 0.0,
                "total_rps": 0.0,
            }
        ]
        self.errors = []
        self.total_rps = 0
        self.total_tps = 0
        self.active_users = 0
        self.ttft = 0
        self.log_dict = {}

    def update(self, log_dict: dict) -> None:
        self.run_state = STATE_RUNNING
        if "requests_per_second" in log_dict:
            self.total_rps = log_dict["requests_per_second"]
            self.stats[-1]["total_rps"] = log_dict["requests_per_second"]
            self.stats[-1]["current_rps"] = log_dict["requests_per_second"]
        if "active_users" in log_dict:
            self.active_users = log_dict["active_users"]
        self.log_dict.update(log_dict)

        # Update the charts once we have metrics
        self.charts = create_dynamic_charts_config(self.log_dict)

    def get_stats(self) -> None:
        self.history.append(
            {
                "run_number": self.reset_counter,
                "timestamp": time.time(),
                "args": vars(args).copy(),
                "metrics": self.log_dict.copy(),
            }
        )
        return {
            "current_response_time_percentiles": {
                "response_time_percentile_0.5": 0,
                "response_time_percentile_0.95": 0,
            },
            "errors": self.errors,
            "fail_ratio": 0.0,
            "state": self.run_state,
            "stats": self.stats,
            "total_fail_per_sec": 0,
            "total_rps": self.total_rps,
            "user_count": self.active_users,
            "custom_metrics": self.log_dict,
            "charts": self.charts,
        }


state = GlobalState()


class ProcessManager:
    def __init__(self) -> None:
        self.user_process = None
        self.metrics_queue = None
        self.user_control_queue = None
        self.proceses_running = False
        self.collector = None

    def start_processes(
        self,
        metrics_logging_interval: int,
        quantiles: list[int],
        model_client: BaseModelClient,
        user_addition_count: int,
        user_addition_time: float,
        max_users: int,
    ) -> None:
        if self.user_process and self.user_process.is_alive():
            raise RuntimeError("Process already running")

        self.metrics_queue = Queue()
        self.user_control_queue = Queue()

        self.user_process = Process(
            target=start_user_loop_async,
            args=(
                model_client,
                self.metrics_queue,
                self.user_control_queue,
                user_addition_count,
                user_addition_time,
                max_users,
            ),
        )
        self.user_process.start()

        self.collector = MetricsCollector(
            global_state=state,
            metrics_window_size=metrics_logging_interval,
            quantiles=quantiles,
            metrics_queue=self.metrics_queue,
            model_client=model_client,
        )
        self.collector.start_logging()
        self.proceses_running = True

    def stop_processes(self) -> None:
        if self.user_process and self.user_process.is_alive():
            self.user_control_queue.put(TriggerShutdown())
            time.sleep(5)  # Kill all users after 5 seconds
            logger.info("Killing all users")
            self.user_process.terminate()
            self.user_process.join()
        self.user_process = None
        if self.collector:
            self.collector.stop_logging()
            self.collector = None
        if self.metrics_queue:
            self.metrics_queue.close()
            self.metrics_queue.join_thread()
            self.metrics_queue = None
        self.proceses_running = False


process_manager = ProcessManager()


# Get the absolute path to the dist directory
dist_dir = Path("webui/dist").absolute()

# Setup Jinja2 templates pointing to the dist directory
templates = Jinja2Templates(directory=str(dist_dir))

# Mount static files
app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request) -> HTMLResponse:
    template_args = {
        "available_shape_classes": ["Default"],
        "available_user_classes": ["ExampleUser"],
        "percentiles_to_chart": [0.5, 0.95],
        "extra_options": {},
        "history": [],
        "host": "https://www.example.com",
        "is_distributed": False,
        "is_shape": None,
        "locustfile": "main.py",
        "num_users": None,
        "override_host_warning": False,
        "show_userclass_picker": False,
        "spawn_rate": None,
        "state": "ready",
        "stats_history_enabled": False,
        "tasks": "{}",
        "user_count": 0,
        "version": "2.15.0",
        "worker_count": 0,
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "template_args": template_args,
            "theme": "",  # Optional theme parameter
        },
    )


@app.post("/swarm")
async def start_swarm(
    user_count: int = Form(...),
    spawn_rate: float = Form(...),
    host: str | None = Form(None),
    tokenizer: str | None = Form(None),
    model: str | None = Form(None),
    max_tokens: int | None = Form(None),
    prompt_min_tokens: int | None = Form(None),
    prompt_max_tokens: int | None = Form(None),
    use_random_prompts: bool | None = Form(None),
    use_single_prompt: bool | None = Form(None),
    ignore_eos: bool | None = Form(None),
    openai_api_key: str | None = Form(None),
) -> dict:
    # Update args with form values if provided
    update_args(
        tokenizer=tokenizer,
        model=model,
        max_tokens=max_tokens,
        prompt_min_tokens=prompt_min_tokens,
        prompt_max_tokens=prompt_max_tokens,
        use_random_prompts=use_random_prompts,
        use_single_prompt=use_single_prompt,
        ignore_eos=ignore_eos,
        user_count=user_count,
        host=host,
        spawn_rate=spawn_rate,
    )

    user_addition_time = 1 / spawn_rate if spawn_rate > 0 else 0
    if not process_manager.proceses_running:
        # Create the tokenizer and prompts
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if not tokenizer.chat_template:
            tokenizer.chat_template = "{{prompt}}"
        if args.use_random_prompts:
            prompts = get_prompt_set_random(tokenizer)
        elif args.use_single_prompt:
            prompts = get_prompt_set_single(tokenizer)
        else:
            prompts = get_prompt_set(
                tokenizer, args.prompt_min_tokens, args.prompt_max_tokens
            )

        model_client = OpenAIChatStreamingClient(
            base_url=args.host.rstrip("/"),
            prompts=prompts,
            system_prompt=system_prompt
            if not (args.use_random_prompts or args.use_single_prompt)
            else None,
            openai_model_name=args.model,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            seed=args.seed,
            ignore_eos=args.ignore_eos,
            openai_api_key=openai_api_key,
        )
        process_manager.start_processes(
            args.metrics_logging_interval,
            args.quantiles,
            model_client,
            user_addition_count=1,
            user_addition_time=user_addition_time,
            max_users=user_count,
        )
    else:
        process_manager.user_control_queue.put(
            SetUserInfo(
                max_users=user_count,
                user_addition_count=1,
                user_addition_time=user_addition_time,
            )
        )
    state.run_state = STATE_SPAWNING
    state.start_time = time.time()

    return {
        "success": True,
        "message": f"Swarming started with {user_count} users, spawn rate: {spawn_rate}",
    }


@app.get("/stop")
async def stop_test() -> dict:
    try:
        if state.run_state not in [STATE_RUNNING, STATE_SPAWNING]:
            raise HTTPException(status_code=400, detail="No test is running")

        process_manager.stop_processes()
        state.reset()
        state.run_state = STATE_STOPPED
        return {"success": True, "message": "Load test stopped"}
    except Exception as e:
        state.errors.append(str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/stats/requests")
async def get_stats() -> JSONResponse:
    return JSONResponse(state.get_stats())


@app.get("/stats/reset")
async def reset_stats() -> dict:
    state.reset()
    return {"success": True}


@app.get("/exceptions")
async def get_exceptions() -> dict:
    return {"exceptions": state.errors}


@app.get("/logs")
async def get_logs() -> dict:
    if not Path("app.log").exists():
        return {"master": [], "workers": {}}

    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(
        None, lambda: Path("app.log").read_text().splitlines()
    )
    return {"master": content, "workers": {}}


@app.get("/stats/requests/csv")
async def get_stats_csv() -> StreamingResponse:
    """Download state history as CSV file."""
    if not state.history:
        raise HTTPException(status_code=404, detail="No history data available")

    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)

    metrics_keys = set()

    # Write header
    for record in state.history:
        metrics_keys.update(record["metrics"].keys())

    writer.writerow(["run_number", "timestamp", "args"] + list(metrics_keys))

    # Write data
    for record in state.history:
        row = [
            record["run_number"],
            record["timestamp"],
            str(record["args"]),
        ]
        row.extend(record["metrics"].get(key, "") for key in metrics_keys)
        writer.writerow(row)

    # Prepare response
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=llm_benchmark_history_{int(time.time())}.csv"
        },
    )


# Setup loggers
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Start FastAPI
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8089,
        log_level="info",
        log_config="logging.conf",
        use_colors=True,
    )
