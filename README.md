# EAL Learning Companion

A Streamlit app that helps English as an Additional Language (EAL) student read an academic passage. Paste up to 4,000 characters, choose a translation language and an English level, then generate:

- Simplified English at CEFR A2, B1, or B2.
- A translation of the **original** passage.
- Up to five vocabulary cards with simple English meanings and translated words and meanings.
- Three comprehension questions with suggested answers.

You can optionally list key terms that must stay unchanged in the simplified text. Each term must appear exactly in the passage. Simplified text and translations have copy buttons. The layout works on desktop and phones.

Translation languages: Arabic, Simplified and Traditional Chinese, French, German, Japanese, Polish, Portuguese, Russian, Spanish, Thai, Turkish, and Urdu.

## Run locally

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with your OpenAI API key:

```toml
OPENAI_API_KEY = "your-key-here"
```

The secrets file is ignored by Git. Start the app with `streamlit run app.py` and open the local URL shown in the terminal (normally port 8501).

## Model and limits

The app calls OpenAI `gpt-6-luna` and checks responses against a strict JSON Schema plus local content rules. An incomplete result may trigger one retry. Each model attempt has a 40-second timeout.

Each browser session gets 20 generation requests and at most three requests per minute. Across sessions on one app host, the app allows 200 model calls per UTC day and 30 per minute. Retries count toward the shared allowance. Counts are stored in `.eal_helper_usage.sqlite3`; set `EAL_USAGE_DB_PATH` to use another location. A deployment on multiple hosts needs a shared store or provider-side spending limit for a deployment-wide cap.

Language and English level selections last for the current browser session. AI simplification and translation can miss nuance, so students should check important details with a teacher or the source text.
