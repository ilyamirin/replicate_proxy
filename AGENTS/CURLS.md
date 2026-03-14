# Curl Checks

## List models

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

## Echo model

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "echo",
    "messages": [
      {"role": "system", "content": "Reply with the user message."},
      {"role": "user", "content": "echo smoke test"}
    ]
  }'
```

## GPT-5.4 with model options

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "reasoning_effort": "high",
    "verbosity": "medium",
    "max_completion_tokens": 200,
    "messages": [
      {"role": "system", "content": "Reply briefly."},
      {"role": "user", "content": "Explain why the sky looks blue."}
    ]
  }'
```

## GPT-5.4 streaming with usage

```bash
curl -N -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "stream": true,
    "stream_options": {"include_usage": true},
    "reasoning_effort": "medium",
    "verbosity": "low",
    "max_completion_tokens": 120,
    "messages": [
      {"role": "system", "content": "Reply briefly."},
      {"role": "user", "content": "Say hello in three different languages."}
    ]
  }'
```

## GPT-5.4 multimodal message

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is shown in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://replicate.delivery/pbxt/JXAkGteY8gT4dRjz0Qexample/cat.png",
              "detail": "high"
            }
          }
        ]
      }
    ]
  }'
```

## GPT-5-nano quick check

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5-nano",
    "reasoning_effort": "low",
    "verbosity": "low",
    "max_completion_tokens": 50,
    "messages": [
      {"role": "system", "content": "Reply briefly."},
      {"role": "user", "content": "What is 12 * 12?"}
    ]
  }'
```
