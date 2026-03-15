# replicate-proxy

Minimal FastAPI server with OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`, including SSE streaming, plus Replicate-backed image tools.

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
Image tools default to `google/nano-banana-2` and `qwen/qwen-image-edit-plus`.

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
- `GET /v1/tools` returns the internal tool list and JSON schema for each tool
- `POST /v1/chat/completions` resolves request `model` through `REPLICATE_MODEL_MAP`
- `POST /v1/chat/completions` routes the local echo model without calling Replicate
- `POST /v1/tools/generate_image` calls `google/nano-banana-2` on Replicate
- `POST /v1/tools/edit_image_uncensored` calls `qwen/qwen-image-edit-plus` on Replicate
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
- The image tool accepts `prompt`, optional `image_input` with up to 14 images, `aspect_ratio`, `resolution`, `google_search`, `image_search`, and `output_format`
- Image tool inputs may be remote URLs, `data:` URLs, existing Replicate file URLs, or local file paths
- Local file paths are restricted to `REPLICATE_LOCAL_IMAGE_INPUT_ROOTS`; by default only `tests/fixtures` and `artifacts/uploads` are allowed
- By default the image tool downloads the generated image into `REPLICATE_IMAGE_OUTPUT_DIR` and returns both `output_url` and `local_path`
- The Qwen edit tool requires `prompt` plus at least one `image_input`
- The Qwen edit tool forwards validated `aspect_ratio`, `go_fast`, `seed`, `output_format`, and `output_quality`
- The Qwen edit tool forces `disable_safety_checker=true` via server config and returns `output_urls` plus local downloaded copies
- Sync mode uses `Prefer: wait=<seconds>`
- If Replicate returns an incomplete non-stream prediction, the app polls the `urls.get` URL until completion or timeout
- Stream errors before the upstream stream starts return normal HTTP `502`
- Stream errors after the stream has started are emitted inside SSE and then end with `[DONE]`

## Image tool example

List tools:

```bash
curl -sS http://127.0.0.1:8000/v1/tools
```

Generate an image:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/tools/generate_image \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "storybook watercolor fox and nightingale at dusk",
    "aspect_ratio": "3:4",
    "resolution": "1K",
    "output_format": "png"
  }'
```

Edit an image without the safety checker:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/tools/edit_image_uncensored \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Turn this comic cover into a watercolor children'\''s book cover.",
    "image_input": ["tests/fixtures/vision-comic.jpeg"],
    "aspect_ratio": "match_input_image",
    "go_fast": true,
    "output_format": "png",
    "output_quality": 95
  }'
```
