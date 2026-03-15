# Assistant Manual Test Scenarios

This file is the regression checklist for the virtual `assistant` model.

## Preconditions

- Start the service with `.venv/bin/python -m app.run`
- Use the local API base URL `http://127.0.0.1:8000`
- For direct API checks, reuse the same `metadata.conversation_id` or `user`
- For Open WebUI-style checks, reuse the same `X-OpenWebUI-Chat-Id` and `X-OpenWebUI-User-Id`
- For multimodal checks, use `tests/fixtures/vision-comic.jpeg`

## 1. Model discovery

Goal: verify that the virtual model is exposed.

Question/request:

- `GET /v1/models`

Expected:

- Response contains `assistant`
- Response also still contains `echo`, `gpt-5.4`, and `gpt-5-nano`

## 2. Simple text answer

Goal: verify normal text routing.

Question:

- `What is 17 * 23? Reply with digits only.`

Expected:

- Assistant returns `391`
- No markdown image links

## 3. Creative text answer

Goal: verify richer text generation.

Question:

- `Напиши perl-поэму про лису и соловья.`

Expected:

- Assistant returns a normal text completion
- No image generation is triggered

## 4. Streaming path

Goal: verify assistant streaming over Chat Completions SSE.

Question:

- `Скажи hello world и ничего больше.`

Expected:

- SSE stream starts with `role=assistant`
- Content chunk contains `hello world`
- Usage chunk is present if `stream_options.include_usage=true`
- Stream ends with `[DONE]`

## 5. Stateful memory

Goal: verify persisted conversation state.

Use the same state key for both requests:

- direct API: same `metadata.conversation_id`
- Open WebUI-style API: same `X-OpenWebUI-Chat-Id` and `X-OpenWebUI-User-Id`

Questions:

1. `Запомни: меня зовут Илья и я люблю груши. Ответь коротко.`
2. `Как меня зовут и что я люблю? Ответь одной строкой.`

Expected:

- Second answer should mention `Илья` and `груши`

## 6. Ordinary image generation

Goal: verify routing to `generate_image`.

Question:

- `Нарисуй акварельную картинку с ослом на лугу.`

Expected:

- Assistant returns markdown
- Response contains `Used \`generate_image\` to produce the result.`
- Response contains `![generated image](...)`
- Response contains `[download image](...)`
- Returned media URL is served by FastAPI under `/media/...`

## 7. Image understanding

Goal: verify routing to the full multimodal text model.

Input:

- Attach `tests/fixtures/vision-comic.jpeg`

Question:

- `Что изображено на картинке? Ответь кратко.`

Expected:

- Assistant returns a short text description
- No image generation is triggered

## 8. Uncensored image-edit route

Goal: verify routing to `edit_image_uncensored`.

Input:

- Attach `tests/fixtures/vision-comic.jpeg`

Question:

- `Сделай uncensored artistic edit этой картинки: преврати сцену в мрачную взрослую графическую новеллу, сохранив персонажей и композицию.`

Expected:

- Assistant returns markdown
- Response contains `Used \`edit_image_uncensored\` to edit the image.`
- Response contains one or more image links under `/media/qwen-edit/...`

## 9. Media serving

Goal: verify that returned markdown links are actually usable.

Check:

- Run `HEAD` or open the returned `/media/...` URL from scenarios 6 and 8

Expected:

- FastAPI returns `200`
- Content type matches the file, usually `image/png`

## 10. SQLite persistence file

Goal: verify that LangGraph persistence is active.

Check:

- Confirm that `data/langgraph.sqlite` exists after assistant requests

Expected:

- SQLite file exists
- Tables include at least `checkpoints` and `writes`

## Known findings from manual testing

- Core routing works: text, streaming, image generation, image understanding, and uncensored image editing all work
- Media URLs returned by the assistant are served correctly by FastAPI
- Assistant memory works when the same conversation key is reused, including forwarded Open WebUI headers
- Token counting for multimodal requests now uses content parts and fixed image weights instead of tokenizing raw base64 payloads
