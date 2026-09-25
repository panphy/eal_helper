# EAL Learning Companion — contributor guide

This is a single-file Streamlit app for EAL students. It simplifies academic passages to CEFR A2, B1, or B2; translates the original passage; and returns vocabulary support and comprehension questions.

## Project map

- `app.py`: UI, session state, usage limits, and OpenAI integration.
- `.streamlit/config.toml`: Streamlit theme. Keep its colors aligned with the CSS tokens in `app.py`.
- `requirements.txt`: Python dependencies. The devcontainer uses Python 3.11.
- `README.md`: setup and user-facing behavior.

## Run and verify

Install `requirements.txt`, put `OPENAI_API_KEY` in `.streamlit/secrets.toml`, then run `streamlit run app.py`. The secrets file is ignored by Git.

There is no automated test suite. For behavior changes, check simplification, translation, vocabulary, questions, protected terms, errors, and usage limits. Check desktop and phone layouts for UI changes. Avoid paid API calls when a local mock can verify the flow.

## Implementation rules

- Add type hints to function signatures. Keep Streamlit UI and session-state operations on the main script thread.
- Use `on_change` callbacks to clear stale results. Language and level preferences stay in the current Streamlit session.
- The API call uses `gpt-6-luna` with strict JSON Schema output and one validation retry. Keep the schema, prompt, and rendered fields in sync.
- Protected terms must appear exactly in the source and simplified text. Vocabulary contains up to five distinct words from the source. Translation quality still needs human review.
- Usage limits are 20 requests and 3 requests per minute per session, plus 200 model calls per UTC day and 30 calls per minute per app host. `reserve_api_call()` charges every attempt in a local SQLite database. Multi-host deployments need a shared limit or provider-side spending control.
- Escape model and user text before placing it in HTML. Keep keyboard focus visible and use the PanPhy ivory, charcoal, and terracotta theme.

Use `panphy/codex/*` feature branches (`panphy/claude/*` for Claude) and the repository's PR review workflow.
