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

## Upload local test image to Replicate Files API

```bash
REPLICATE_FILE_URL=$(
  TOKEN=$(sed -n 's/^REPLICATE_API_TOKEN=//p' .env) && \
  curl -sS -X POST https://api.replicate.com/v1/files \
    -H "Authorization: Bearer $TOKEN" \
    -F content=@tests/fixtures/vision-comic.jpeg | \
  jq -r '.urls.get'
)
```

## GPT-5.4 multimodal via native image_input

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "reasoning_effort": "none",
    "verbosity": "low",
    "max_completion_tokens": 300,
    "prompt": "Describe the image in one short sentence.",
    "image_input": ["'"$REPLICATE_FILE_URL"'"]
  }'
```

## GPT-5.4 multimodal via OpenAI-style messages

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "reasoning_effort": "low",
    "verbosity": "low",
    "max_completion_tokens": 300,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe the image in one short sentence."},
          {
            "type": "image_url",
            "image_url": {
              "url": "'"$REPLICATE_FILE_URL"'",
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

## GPT-5-nano multimodal via native image_input

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5-nano",
    "reasoning_effort": "minimal",
    "verbosity": "low",
    "max_completion_tokens": 300,
    "prompt": "Describe the image in one short sentence.",
    "image_input": ["'"$REPLICATE_FILE_URL"'"]
  }'
```
