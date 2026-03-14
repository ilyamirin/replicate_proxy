# replicate-proxy

Minimal FastAPI server with OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`, including SSE streaming.

## Config

All settings are read from environment variables or a local `.env`.

Copy `.env.example` to `.env` and adjust values if needed.

Direct dependencies are also duplicated in `requirements.txt`.
When a new library is installed for this project, add it there too.

Available public models come from `REPLICATE_MODEL_MAP`.
Local echo is exposed as model `echo` by default and can be renamed with `ECHO_MODEL_ID`.
Settings are loaded once when the service starts.
Token counting uses local `tiktoken` data from `.tiktoken-cache/o200k_base.tiktoken`.
The repository must contain `.tiktoken-cache/o200k_base.tiktoken`; startup copies it to the cache key path expected by `tiktoken`.
Default Replicate models in the config are `gpt-5.4` and `gpt-5-nano`.

## Run

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e '.[test,dev]'
.venv/bin/python -m app.run
```

If `.tiktoken-cache/o200k_base.tiktoken` is missing, the service will fail fast on startup.

## Test

```bash
.venv/bin/pytest
```

## Pre-commit checks

Always run these before commit:

```bash
.venv/bin/ruff format .
.venv/bin/ruff check .
.venv/bin/bandit -c pyproject.toml -r app
.venv/bin/pytest
```

## Replicate notes

- `GET /v1/models` returns the configured public model list
- `POST /v1/chat/completions` resolves request `model` through `REPLICATE_MODEL_MAP`
- `POST /v1/chat/completions` routes the local echo model without calling Replicate
- `stream=true` returns real `text/event-stream` chunks
- Token usage is calculated with local `tiktoken` using `o200k_base`
- `REPLICATE_MODEL_MAP` format: `public-id=owner/model-name`
- Current defaults: `gpt-5.4=openai/gpt-5.4`, `gpt-5-nano=openai/gpt-5-nano`
- Official prediction endpoint pattern: `POST /v1/models/{owner}/{name}/predictions`
- The app always sends `messages`
- Replicate models require the client to send `reasoning_effort`
- The app also forwards OpenAI-style request fields `verbosity` and `max_completion_tokens` when the client provides them
- Optional server-side fallback envs exist only for `verbosity` and `max_completion_tokens`
- The request may use either `messages` or Replicate-native `prompt`, `system_prompt`, and `image_input`
- `messages[].content` may be a plain string or an OpenAI-style content-part array with `text` and `image_url`
- Sync mode uses `Prefer: wait=<seconds>`
- If Replicate returns an incomplete non-stream prediction, the app polls the `urls.get` URL until completion or timeout
- Stream errors before the upstream stream starts return normal HTTP `502`
- Stream errors after the stream has started are emitted inside SSE and then end with `[DONE]`
