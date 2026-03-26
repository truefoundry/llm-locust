# Locust LLM Load Test Server

Distributed Locust setup for load testing an LLM chat completions endpoint. Supports **local runs** (Docker Compose) and **TrueFoundry** (Kubernetes with auto-scaled workers).

## Architecture

- **Master**: Coordinates workers, serves Web UI on `:8089`, aggregates metrics.
- **Workers**: Run `FastHttpUser` for high throughput; scale horizontally.
- **User classes** (select via `--tags` or class weight):
  - **LLMStreamingUser**: Streaming SSE; reports TTFT (time-to-first-token) and stream duration.
  - **LLMThroughputUser**: Non-streaming; maximizes RPS stress.

## Quick start (local — Docker Compose)

1. Copy env and set your API token:
   ```bash
   cp .env.example .env
   # Edit .env: set LOCUST_HOST and LOCUST_AUTH_TOKEN
   ```

2. Start master and workers:
   ```bash
   docker compose up --build --scale worker=10 -d
   ```

3. Open the Web UI at **http://localhost:8089**, set users/spawn rate, and start the test.

4. Stop:
   ```bash
   docker compose down
   ```

## TrueFoundry deployment

1. Install CLI and log in:
   ```bash
   pip install truefoundry
   tfy login --host https://vedantaz.devtest.truefoundry.tech
   ```

2. Set workspace and token in `.env`:
   ```bash
   TFY_WORKSPACE_FQN=your-workspace-fqn
   LOCUST_AUTH_TOKEN=your-api-token
   LOCUST_EXPECT_WORKERS=10   # optional; default 10
   ```

3. Deploy master, then workers:
   ```bash
   python deploy_master.py
   # Wait for master to be healthy, then:
   python deploy_workers.py
   ```

4. Open the master URL from the TrueFoundry dashboard. Workers auto-scale (1–30 replicas) at 90% CPU via HPA.

## Scaling

| Environment   | How to scale |
|---------------|----------------|
| Docker Compose | `docker compose up -d --scale worker=N` |
| TrueFoundry   | HPA: 1–30 workers by CPU (90%). No manual scaling needed. |

## Configuration

- **locust.conf**: Default host, users, spawn rate, web port (reference for local runs).
- **.env**: `LOCUST_HOST`, `LOCUST_AUTH_TOKEN`, `LOCUST_LLM_MODEL`, `LOCUST_MAX_TOKENS`, `LOCUST_EXPECT_WORKERS`, `TFY_WORKSPACE_FQN`.

Run a specific user class (e.g. throughput only):
```bash
locust --host https://your-api.com -f locustfile.py LLMThroughputUser
# or by tag:
locust --host https://your-api.com -f locustfile.py --tags throughput
```
