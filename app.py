import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import re

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="EAL Scaffolder",
    page_icon="🎓",
    layout="wide"
)

# -------------------------
# Helpers
# -------------------------
LANGUAGE_MAP = {
    "Spanish": "Spanish",
    "Polish": "Polish",
    "Arabic": "Arabic",
    "Urdu": "Urdu",
    "Chinese": "Simplified Chinese",
    "French": "French",
    "Japanese": "Japanese",
    "Portuguese": "Portuguese",
}

def extract_cefr(level_label: str) -> str:
    """Convert 'Intermediate (B1)' -> 'B1' (fallback: original string)."""
    m = re.search(r"\((A1|A2|B1|B2|C1|C2)\)", level_label)
    return m.group(1) if m else level_label

@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def safe_get_str(d: dict, key: str, default: str = "") -> str:
    v = d.get(key, default)
    return v if isinstance(v, str) else default

def safe_get_list(d: dict, key: str) -> list:
    v = d.get(key, [])
    return v if isinstance(v, list) else []

# -------------------------
# Sidebar (Settings)
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("🚨 Admin Error: OpenAI API Key not found in secrets.")
        st.stop()

    st.divider()

    target_lang_ui = st.selectbox(
        "Translation Language",
        list(LANGUAGE_MAP.keys()),
        index=0
    )
    target_lang = LANGUAGE_MAP[target_lang_ui]

    difficulty_label = st.select_slider(
        "Simplification Level",
        options=["Beginner (A2)", "Intermediate (B1)", "Advanced (B2)"],
        value="Intermediate (B1)"
    )
    cefr = extract_cefr(difficulty_label)

    vocab_n = st.slider("Number of vocabulary words", 3, 12, 5)

    translate_mode = st.radio(
        "Translate which text?",
        options=[
            "Translate the simplified text (recommended)",
            "Translate the original text"
        ],
        index=0
    )

# -------------------------
# Main page
# -------------------------
st.title("🎓 Academic Text Helper")
st.markdown(
    "Paste your difficult text below to get a simplified version, a full translation, and a translated vocabulary list."
)

source_text = st.text_area(
    "Paste text here:",
    height=200,
    placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy..."
)

# -------------------------
# AI function
# -------------------------
def get_scaffolded_content(text: str, language: str, cefr_level: str, n_vocab: int, mode: str) -> dict | None:
    client = get_client(api_key)

    translate_target = "SIMPLIFIED_TEXT" if "simplified" in mode.lower() else "ORIGINAL_TEXT"

    prompt = f"""
You are an expert EAL teacher.

You will receive INPUT_TEXT.

Tasks:
1) Simplify INPUT_TEXT into clear academic English at CEFR {cefr_level}. Preserve meaning.
   Output this as SIMPLIFIED_TEXT.

2) Translate:
   If TRANSLATE_TARGET is SIMPLIFIED_TEXT, translate SIMPLIFIED_TEXT into {language}.
   If TRANSLATE_TARGET is ORIGINAL_TEXT, translate INPUT_TEXT into {language}.
   Output this as TRANSLATED_TEXT.
   TRANSLATED_TEXT must be entirely in {language}. Do not include English.

3) Vocabulary:
   Pick exactly {n_vocab} difficult academic words from INPUT_TEXT (not random).
   For each word provide:
   - definition: simple English definition
   - translation: direct translation into {language}

Return JSON ONLY with EXACTLY this shape:
{{
  "simplified_text": "....",
  "translated_text": "....",
  "vocabulary": [
    {{
      "word": "English word",
      "definition": "simple English definition",
      "translation": "the {language} translation"
    }}
  ]
}}

Rules:
- No markdown.
- No extra keys.
- vocabulary must contain exactly {n_vocab} items.
- Keep definitions simple and student-friendly.

TRANSLATE_TARGET: {translate_target}

INPUT_TEXT:
{text}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You output strict JSON only. No extra keys, no markdown."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Minimal validation and cleanup
        if not isinstance(data, dict):
            raise ValueError("Model returned non-object JSON.")

        simplified = safe_get_str(data, "simplified_text")
        translated = safe_get_str(data, "translated_text")
        vocab = safe_get_list(data, "vocabulary")

        # Normalize vocabulary rows
        cleaned_vocab = []
        for item in vocab:
            if not isinstance(item, dict):
                continue
            cleaned_vocab.append({
                "word": safe_get_str(item, "word"),
                "definition": safe_get_str(item, "definition"),
                "translation": safe_get_str(item, "translation"),
            })

        # If model returned more or fewer, trim/pad gently to keep UI stable
        if len(cleaned_vocab) > n_vocab:
            cleaned_vocab = cleaned_vocab[:n_vocab]
        while len(cleaned_vocab) < n_vocab:
            cleaned_vocab.append({"word": "", "definition": "", "translation": ""})

        return {
            "simplified_text": simplified,
            "translated_text": translated,
            "vocabulary": cleaned_vocab
        }

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# -------------------------
# Action button
# -------------------------
if st.button("✨ Simplify & Translate", type="primary"):
    if not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner(f"Simplifying (CEFR {cefr}) and translating to {target_lang_ui}..."):
            data = get_scaffolded_content(
                text=source_text.strip(),
                language=target_lang,
                cefr_level=cefr,
                n_vocab=vocab_n,
                mode=translate_mode
            )

        if data:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📖 Simplified English")
                st.success(data["simplified_text"] if data["simplified_text"] else "(No simplified text returned)")

            with col2:
                st.subheader(f"🌍 Translation ({target_lang_ui})")
                st.info(data["translated_text"] if data["translated_text"] else "(No translation returned)")

            st.divider()
            st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

            df = pd.DataFrame(data["vocabulary"])

            # Force the columns and order so the translation always shows
            for c in ["word", "definition", "translation"]:
                if c not in df.columns:
                    df[c] = ""
            df = df[["word", "definition", "translation"]].rename(columns={
                "word": "Word",
                "definition": "English Definition",
                "translation": f"Translation ({target_lang_ui})"
            })

            st.dataframe(df, hide_index=True, use_container_width=True)