# replicate-proxy

Minimal FastAPI server with an OpenAI-compatible `POST /v1/chat/completions` endpoint.

## Config

All settings are read from environment variables or a local `.env`.

Copy `.env.example` to `.env` and adjust values if needed.

## Run

```bash
.venv/bin/pip install -e .[test]
.venv/bin/python -m app.run
```

## Test

```bash
.venv/bin/pytest
```
