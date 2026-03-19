# AGENTS README

## Project summary

- Project: `replicate-proxy`
- Purpose: minimal FastAPI server with OpenAI-compatible `GET /v1/models`, `POST /v1/chat/completions`, a stateful LangGraph assistant model, and Replicate-backed image tools
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
- `app/assistant_graph.py`: LangGraph orchestration model with persisted state
- `app/backends.py`: startup-time service container
- `app/clients/replicate.py`: Replicate HTTP client with sync, polling, and SSE stream support
- `app/clients/replicate_images.py`: Replicate client for image generation/editing tools
- `app/clients/replicate_qwen_edit.py`: Replicate client for `qwen/qwen-image-edit-plus`
- `app/config.py`: settings loader from env and `.env`, loaded once at service start
- `app/services.py`: echo business logic
- `app/schemas.py`: request/response models
- `app/tool_schemas.py`: internal tool request/response models
- `app/tokens.py`: token counting via local `tiktoken` cache using `o200k_base`
- `app/run.py`: entrypoint that starts uvicorn using env-based host/port
- `tests/test_api.py`: API and config tests
- `AGENTS/ASSISTANT_TESTS.md`: manual regression scenarios for the virtual assistant model
- `AGENTS/CLARIFICATION_TESTS.md`: planned unit/manual scenarios for the future clarification and guard layer
- `.env.example`: local configuration template
- `requirements.txt`: explicit dependency list that must be kept in sync with installs
- `pyproject.toml`: dependencies and pytest config

## Environment variables

- `APP_NAME`: FastAPI title
- `APP_HOST`: host for uvicorn
- `APP_PORT`: port for uvicorn
- `APP_API_PREFIX`: API prefix, default `/v1`
- `APP_HEALTH_PATH`: health endpoint path, default `/health`
- `APP_PUBLIC_BASE_URL`: externally reachable base URL used in markdown/media links
- `APP_MEDIA_PATH`: static media path, default `/media`
- `APP_MEDIA_ROOT`: local root served by FastAPI StaticFiles, default `artifacts`
- `APP_ECHO_EMPTY_RESPONSE`: fallback assistant text when there is no user message
- `ECHO_MODEL_ID`: public model id for the local echo backend, default `echo`
- `ASSISTANT_MODEL_ID`: public id for the stateful LangGraph assistant model, default `assistant`
- `ASSISTANT_ROUTER_MODEL_ID`: router LLM for assistant orchestration, default `gpt-5-nano`
- `ASSISTANT_FULL_MODEL_ID`: heavier multimodal text model used by assistant when needed, default `gpt-5.4`
- `ASSISTANT_SQLITE_PATH`: local SQLite path for LangGraph checkpoints, default `data/langgraph.sqlite`
- `REPLICATE_API_TOKEN`: local secret, only in `.env`
- `REPLICATE_BASE_URL`: default `https://api.replicate.com/v1`
- `REPLICATE_MODEL_MAP`: comma-separated `public-id=owner/model-name` entries used by `/v1/models` and chat routing
  Current defaults include `gpt-5.4` and `gpt-5-nano`
- `REPLICATE_IMAGE_TOOL_ID`: public id for the internal image tool, default `generate_image`
- `REPLICATE_IMAGE_TOOL_MODEL`: target Replicate image model, default `google/nano-banana-2`
- `REPLICATE_IMAGE_OUTPUT_DIR`: local output directory for downloaded generated images
- `REPLICATE_IMAGE_DOWNLOAD_OUTPUT`: if `true`, download generated outputs locally and return `local_path`
- `REPLICATE_QWEN_EDIT_TOOL_ID`: public id for the Qwen edit tool, default `edit_image_uncensored`
- `REPLICATE_QWEN_EDIT_TOOL_MODEL`: target Replicate edit model, default `qwen/qwen-image-edit-plus`
- `REPLICATE_QWEN_EDIT_OUTPUT_DIR`: local output directory for downloaded Qwen edit results
- `REPLICATE_QWEN_EDIT_DOWNLOAD_OUTPUT`: if `true`, download Qwen outputs locally and return `local_paths`
- `REPLICATE_QWEN_EDIT_FORCE_DISABLE_SAFETY_CHECKER`: if `true`, always send `disable_safety_checker=true` upstream
- `REPLICATE_LOCAL_IMAGE_INPUT_ROOTS`: comma-separated allowlist for local `image_input` paths, default `tests/fixtures,artifacts/uploads`
- `REPLICATE_DEFAULT_VERBOSITY`: optional fallback for requests that omit `verbosity`
- `REPLICATE_DEFAULT_MAX_COMPLETION_TOKENS`: optional fallback for requests that omit `max_completion_tokens`
  For `claude-4.5-sonnet`, this fallback must stay within `1024..64000`.
- `REPLICATE_SYNC_WAIT_SECONDS`: sync wait header for Replicate
- `REPLICATE_POLL_INTERVAL_SECONDS`: poll interval after incomplete sync response
- `REPLICATE_POLL_TIMEOUT_SECONDS`: max poll duration
- `REPLICATE_HTTP_TIMEOUT_SECONDS`: HTTP timeout
- `REPLICATE_TRANSPORT_RETRIES`: retry count for transport-level Replicate failures
- `REPLICATE_TRANSPORT_RETRY_BACKOFF_SECONDS`: exponential backoff base for transport retries

## Important behavior

- `GET {APP_API_PREFIX}/models` returns the configured public model list
- `GET {APP_API_PREFIX}/tools` returns the internal tool list and JSON schema
- `GET {APP_MEDIA_PATH}/...` serves local generated media files
- `POST {APP_API_PREFIX}/chat/completions` returns OpenAI-style JSON
- `POST {APP_API_PREFIX}/chat/completions` with `ASSISTANT_MODEL_ID` runs the LangGraph assistant
- `POST {APP_API_PREFIX}/tools/{REPLICATE_IMAGE_TOOL_ID}` runs `google/nano-banana-2`
- `POST {APP_API_PREFIX}/tools/{REPLICATE_QWEN_EDIT_TOOL_ID}` runs `qwen/qwen-image-edit-plus`
- If the request `model` equals `ECHO_MODEL_ID`, reply content is the last `user` message from the request
- All other known models are resolved via `REPLICATE_MODEL_MAP` and sent to Replicate
- `stream=true` returns OpenAI-style SSE chunks
- OpenAI GPT models require request-level `reasoning_effort`
- `claude-4.5-sonnet` does not use `reasoning_effort` or `verbosity`; it accepts only Claude-compatible fields and at most one image
- Request schema also accepts `verbosity` and `max_completion_tokens`
- If those request fields are absent, the app uses `REPLICATE_DEFAULT_*` env fallbacks only for `verbosity` and `max_completion_tokens`
- Chat request schema also accepts Replicate-native `prompt`, `system_prompt`, and `image_input`
- If there is no `user` message, the response body uses `APP_ECHO_EMPTY_RESPONSE`
- Token usage is calculated with local `tiktoken`
- The repo must include `.tiktoken-cache/o200k_base.tiktoken`; startup derives the hashed cache entry from it
- Request schema accepts `role` in `system|user|assistant|tool`
- `messages[].content` supports either a plain string or an OpenAI-style part array with `text` and `image_url`
- Unknown extra request fields are ignored by pydantic models
- The assistant model uses LangGraph with SQLite persistence
- Assistant state key comes from `metadata.conversation_id`; if absent, falls back to `user`
- Open WebUI integration should use `ENABLE_FORWARD_USER_INFO_HEADERS=true`
- If Open WebUI forwards `X-OpenWebUI-Chat-Id` and `X-OpenWebUI-User-Id`, the service maps them to assistant state automatically
- Assistant returns markdown with embedded/generated media URLs served by this FastAPI app
- The image tool accepts `prompt`, optional `image_input`, `aspect_ratio`, `resolution`, `google_search`, `image_search`, and `output_format`
- Image tool inputs may be remote URLs, `data:` URLs, Replicate file URLs, or local file paths
- Local file paths are allowed only under `REPLICATE_LOCAL_IMAGE_INPUT_ROOTS`
- Generated image outputs on Replicate are short-lived; default behavior is to download them locally into `REPLICATE_IMAGE_OUTPUT_DIR`
- `generate_image` is the default general-purpose image tool
- The Qwen edit tool requires `prompt` and `image_input`
- The Qwen edit tool forwards validated `aspect_ratio`, `go_fast`, `seed`, `output_format`, and `output_quality`
- The Qwen edit tool is intentionally provider-specific: server config forces `disable_safety_checker=true`
- `edit_image_uncensored` is reserved for the uncensored editing path; normal image tasks should stay on `generate_image`
- Qwen edit responses return `output_urls` and `local_paths`

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
- If `APP_PUBLIC_BASE_URL`, `APP_MEDIA_PATH`, or `APP_MEDIA_ROOT` changes, restart the process to apply the new media links/mount
- Existing tests call default paths `/health` and `/v1/chat/completions`
- This is intentionally minimal: no auth on our API and only local persistence for assistant state/artifacts
- `artifacts/` is gitignored and used for downloaded image tool outputs
- `data/` is gitignored and used for LangGraph SQLite state
- Streaming preflight failures should return HTTP `502`; mid-stream failures can only surface inside SSE
- Manual assistant smoke/regression scenarios live in `AGENTS/ASSISTANT_TESTS.md`
- Clarification/guard rollout scenarios and curl checks live in `AGENTS/CLARIFICATION_TESTS.md`
- Replicate clients now retry transport-level failures before surfacing a final `502`

## Good next steps if the project grows

- Add auth via bearer token from env
- Split test fixtures from endpoint tests
- Add artifact cleanup/retention for downloaded image outputs
