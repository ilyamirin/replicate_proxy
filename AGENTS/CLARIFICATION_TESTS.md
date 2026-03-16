# Clarification Tests

These are manual regression scenarios for the clarification/guard layer.
The current implementation is a generic uncertainty-reduction flow:
planner -> guard -> clarify -> resume. The image-selection case is only the
first concrete resource-selection use case on top of that shared mechanism.

## Goals

- The assistant must not silently apply a request to every matching file.
- The assistant should ask one short clarification question when the target file, action, or scope is ambiguous.
- After the user answers, the assistant should resume the original task instead of starting a new one.

## Current V1 behavior

- Planner always runs before execution; assistant forced routes are removed.
- If planner or guard sees unresolved ambiguity, assistant asks one short clarification question instead of guessing.
- If a request targets several candidate resources, assistant asks which resource to use.
- If planner reports a missing required parameter, assistant asks only for that parameter.
- If planner confidence is very low on a vague request, assistant asks what kind of action is actually needed.
- Answers like `1`, `вторая`, and `последняя` resume the original request.
- Ambiguous replies like `ну вот ту` re-ask the same clarification question.
- `отмени` clears the pending clarification and asks for a new request.

## Unit-test coverage targets

- Planner always runs before execution; no forced routes for assistant requests.
- Ambiguous single-target requests produce a clarification question instead of execution.
- Low-confidence vague requests produce an intent clarification instead of guesswork.
- Missing required params produce a parameter clarification and then resume execution.
- Clarification answers like `1`, `первая`, `вторая`, `последняя`, and exact filenames resolve correctly.
- Pending clarification is stored in state and resumed on the next user turn.
- Failed clarification parsing re-asks the same question and does not execute the tool/model.
- Clarification state does not leak into unrelated later turns.

## Manual curl scenarios

Use the same `X-OpenWebUI-Chat-Id` / `X-OpenWebUI-User-Id` across multi-step scenarios.

### 1. Multiple images, OCR request

Step 1:
- Send one request with two images and text `Что написано на картинке?`

Expected:
- No OCR result yet
- Assistant asks which image to use: `1`, `2`, or equivalent

Step 2:
- Reply `На второй.`

Expected:
- Assistant resumes the original OCR task
- Only the second image is analyzed

### 2. Current image vs previous image

Step 1:
- Upload an image, then later upload another image
- Ask `Опиши эту картинку`

Expected:
- If the reference is ambiguous, assistant asks which one to use
- If the current message contains exactly one image, assistant uses that one only

### 3. Assistant-generated image vs original image

Step 1:
- Ask assistant to generate an image
- Then upload a new source image
- Ask `Сделай это в стиле аниме`

Expected:
- Assistant asks whether to edit the generated image or the uploaded source image

### 4. Audio transcription

Step 1:
- Upload two audio files
- Ask `Расшифруй это`

Expected:
- Assistant asks which audio file to transcribe

### 5. XLS processing

Step 1:
- Upload `sales_q1.xlsx` and `sales_q2.xlsx`
- Ask `Найди аномалии в отчете`

Expected:
- Assistant asks which spreadsheet to analyze or whether to process both

### 6. DOC summary

Step 1:
- Upload two documents
- Ask `Сделай summary документа`

Expected:
- Assistant asks which document is meant

### 7. Data-source disambiguation

Step 1:
- Ask `Найди последние цифры по продажам`

Expected:
- If multiple data sources are available, assistant asks where to search:
  - web
  - MongoDB
  - ClickHouse

### 8. Resume after clarification

Step 1:
- Trigger any clarification

Step 2:
- Reply with a short answer like `1` or `последняя`

Expected:
- Assistant resumes the pending task
- It does not treat the short answer as a brand-new user intent

### 9. Failed clarification parsing

Step 1:
- Trigger clarification

Step 2:
- Reply with something still ambiguous like `ну вот ту`

Expected:
- Assistant asks again, more explicitly
- It still does not execute the task

### 10. Cancellation

Step 1:
- Trigger clarification

Step 2:
- Reply `отмени`

Expected:
- Pending clarification is cleared
- Original task is not executed
