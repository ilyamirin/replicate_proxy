# AGENTS README

## Project summary

- Project: `replicate-proxy`
- Purpose: minimal FastAPI server with OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`
- Current backends: Replicate-backed models plus a local echo model
- Python version: `3.12`
- Package/dependency management: `pyproject.toml` + local `.venv`

## Non-negotiable working rules

- Work only inside the local virtual environment: `.venv`
- Do not install or run project dependencies globally
- Configure the app only through environment variables or local `.env`
- Do not commit `.env`, secrets, virtualenv files, caches, or build artifacts
- Before every commit, run the formatter, Python quality linters, Python security linting, and tests
- If a new library is installed, add it to `requirements.txt` immediately

## Current project layout

- `app/main.py`: FastAPI app and HTTP endpoints
- `app/backends.py`: startup-time service container
- `app/clients/replicate.py`: Replicate HTTP client with sync, polling, and SSE stream support
- `app/config.py`: settings loader from env and `.env`, loaded once at service start
- `app/services.py`: echo business logic
- `app/schemas.py`: request/response models
- `app/tokens.py`: token counting via local `tiktoken` cache using `o200k_base`
- `app/run.py`: entrypoint that starts uvicorn using env-based host/port
- `tests/test_api.py`: API and config tests
- `.env.example`: local configuration template
- `requirements.txt`: explicit dependency list that must be kept in sync with installs
- `pyproject.toml`: dependencies and pytest config

## Environment variables

- `APP_NAME`: FastAPI title
- `APP_HOST`: host for uvicorn
- `APP_PORT`: port for uvicorn
- `APP_API_PREFIX`: API prefix, default `/v1`
- `APP_HEALTH_PATH`: health endpoint path, default `/health`
- `APP_ECHO_EMPTY_RESPONSE`: fallback assistant text when there is no user message
- `ECHO_MODEL_ID`: public model id for the local echo backend, default `echo`
- `REPLICATE_API_TOKEN`: local secret, only in `.env`
- `REPLICATE_BASE_URL`: default `https://api.replicate.com/v1`
- `REPLICATE_MODEL_MAP`: comma-separated `public-id=owner/model-name` entries used by `/v1/models` and chat routing
  Current defaults include `gpt-5.4` and `gpt-5-nano`
- `REPLICATE_DEFAULT_REASONING_EFFORT`: optional fallback for requests that omit `reasoning_effort`
- `REPLICATE_DEFAULT_VERBOSITY`: optional fallback for requests that omit `verbosity`
- `REPLICATE_DEFAULT_MAX_COMPLETION_TOKENS`: optional fallback for requests that omit `max_completion_tokens`
- `REPLICATE_SYNC_WAIT_SECONDS`: sync wait header for Replicate
- `REPLICATE_POLL_INTERVAL_SECONDS`: poll interval after incomplete sync response
- `REPLICATE_POLL_TIMEOUT_SECONDS`: max poll duration
- `REPLICATE_HTTP_TIMEOUT_SECONDS`: HTTP timeout

## Important behavior

- `GET {APP_API_PREFIX}/models` returns the configured public model list
- `POST {APP_API_PREFIX}/chat/completions` returns OpenAI-style JSON
- If the request `model` equals `ECHO_MODEL_ID`, reply content is the last `user` message from the request
- All other known models are resolved via `REPLICATE_MODEL_MAP` and sent to Replicate
- `stream=true` returns OpenAI-style SSE chunks
- Request schema also accepts OpenAI-style `reasoning_effort`, `verbosity`, and `max_completion_tokens`
- If those request fields are absent, the app uses `REPLICATE_DEFAULT_*` env fallbacks when configured
- If there is no `user` message, the response body uses `APP_ECHO_EMPTY_RESPONSE`
- Token usage is calculated with local `tiktoken`
- The repo must include `.tiktoken-cache/o200k_base.tiktoken`; startup derives the hashed cache entry from it
- Request schema accepts `role` in `system|user|assistant|tool`
- `messages[].content` supports either a plain string or an OpenAI-style part array with `text` and `image_url`
- Unknown extra request fields are ignored by pydantic models

## Local setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e '.[test,dev]'
cp .env.example .env
```

## Run

```bash
.venv/bin/python -m app.run
```

Default local URL:

```text
http://127.0.0.1:8000
```

## Test

```bash
.venv/bin/pytest
```

## Required pre-commit routine

Run these every time before creating a commit:

```bash
.venv/bin/ruff format .
.venv/bin/ruff check .
.venv/bin/bandit -c pyproject.toml -r app
.venv/bin/pytest
```

What they cover:

- `ruff format .`: autoformat code
- `ruff check .`: Python quality linting and import ordering
- `bandit -c pyproject.toml -r app`: Python security linting
- `pytest`: regression check

## Git and safety

- Local git repo is initialized in this directory
- First commit already exists: `Initial FastAPI OpenAI-compatible server`
- `.gitignore` already excludes `.env`, `.venv`, caches, build outputs, and egg-info
- Secret scan was already run with `gitleaks` and found no leaks at that time
- Before committing new changes, run formatter, linters, tests, and re-check for accidental secrets in `.env`, docs, or copied curl examples
- Never print or commit the real `REPLICATE_API_TOKEN`

## Current limitations and reminders

- The app loads settings once when `create_app()` runs
- To test different config, create a fresh app with explicit `Settings`
- If `APP_API_PREFIX` or `APP_HEALTH_PATH` changes, restart the process to apply the new routes
- Existing tests call default paths `/health` and `/v1/chat/completions`
- This is intentionally minimal: no auth on our API and no persistence
- Streaming preflight failures should return HTTP `502`; mid-stream failures can only surface inside SSE

## Good next steps if the project grows

- Add auth via bearer token from env
- Split test fixtures from endpoint tests
- Add linting and formatting tools
