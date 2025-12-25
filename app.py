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
# Alphabetical by UI label
LANGUAGE_MAP = {
    "Arabic": "Arabic",
    "Chinese (Simplified)": "Simplified Chinese",
    "Chinese (Traditional)": "Traditional Chinese",
    "French": "French",
    "German": "German",
    "Japanese": "Japanese",
    "Polish": "Polish",
    "Portuguese": "Portuguese",
    "Spanish": "Spanish",
    "Thai": "Thai",
    "Turkish": "Turkish",
    "Urdu": "Urdu",
}

LEVEL_OPTIONS = [
    "Beginner (A2)",
    "Intermediate (B1)",
    "Advanced (B2)"
]

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
# API Key check (silent)
# -------------------------
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 Admin Error: OpenAI API Key not found in secrets.")
    st.stop()

client = get_client(api_key)

# -------------------------
# Main UI
# -------------------------
st.title("🎓 Academic Text Helper")
st.markdown("Paste your text to get a simplified version, a full translation of the original text, and a translated vocabulary list.")

# Controls above the input box (no sidebar)
col_ctrl1, col_ctrl2 = st.columns([1, 1])

with col_ctrl1:
    target_lang_ui = st.selectbox(
        "Translation Language",
        list(LANGUAGE_MAP.keys()),
        index=0
    )
    target_lang = LANGUAGE_MAP[target_lang_ui]

with col_ctrl2:
    level_label = st.selectbox(
        "English Level (Simplification)",
        LEVEL_OPTIONS,
        index=1
    )
    cefr = extract_cefr(level_label)

# Side-by-side input and simplified output
col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("📝 Input Text")
    source_text = st.text_area(
        "",
        height=240,
        placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy..."
    )

with col_out:
    st.subheader("📖 Simplified Text (English)")
    simplified_placeholder = st.empty()

# Full translation area (under the side-by-side boxes)
st.subheader(f"🌍 Full Translation of Input Text ({target_lang_ui})")
translation_placeholder = st.empty()

# -------------------------
# AI function
# -------------------------
def get_scaffolded_content(text: str, language: str, cefr_level: str) -> dict | None:
    """
    Returns:
    - simplified_text (English, simplified to CEFR)
    - full_translation (translation of ORIGINAL input text)
    - vocabulary: list of
        {word, definition, translation_word, translation_definition}
    """
    prompt = f"""
You are an expert EAL teacher and a careful translator.

You will receive INPUT_TEXT.

Tasks:
1) Simplify INPUT_TEXT into clear, student-friendly academic English at CEFR {cefr_level}.
   Preserve meaning and keep subject keywords accurate. Output as SIMPLIFIED_TEXT.

2) Translate the ORIGINAL INPUT_TEXT into {language}. Output as FULL_TRANSLATION.
   FULL_TRANSLATION must be entirely in {language} and natural (not word-for-word).

3) Vocabulary:
   Pick exactly 5 difficult academic words from INPUT_TEXT (not random).
   For each word provide:
   - definition: simple English definition
   - translation_word: direct translation of the word into {language}
   - translation_definition: translation of the definition into {language}

Return JSON ONLY with EXACTLY this shape:
{{
  "simplified_text": "....",
  "full_translation": "....",
  "vocabulary": [
    {{
      "word": "English word",
      "definition": "simple English definition",
      "translation_word": "word in {language}",
      "translation_definition": "definition translated into {language}"
    }}
  ]
}}

Rules:
- No markdown.
- No extra keys.
- vocabulary must contain exactly 5 items.
- FULL_TRANSLATION must contain no English.
- translation_definition must contain no English.

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

        if not isinstance(data, dict):
            raise ValueError("Model returned non-object JSON.")

        simplified = safe_get_str(data, "simplified_text")
        full_translation = safe_get_str(data, "full_translation")
        vocab = safe_get_list(data, "vocabulary")

        cleaned_vocab = []
        for item in vocab:
            if not isinstance(item, dict):
                continue
            cleaned_vocab.append({
                "word": safe_get_str(item, "word"),
                "definition": safe_get_str(item, "definition"),
                "translation_word": safe_get_str(item, "translation_word"),
                "translation_definition": safe_get_str(item, "translation_definition"),
            })

        # Hard enforce 5 rows for stable UI
        cleaned_vocab = cleaned_vocab[:5]
        while len(cleaned_vocab) < 5:
            cleaned_vocab.append({
                "word": "",
                "definition": "",
                "translation_word": "",
                "translation_definition": ""
            })

        return {
            "simplified_text": simplified,
            "full_translation": full_translation,
            "vocabulary": cleaned_vocab
        }

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# -------------------------
# Action button
# -------------------------
st.markdown("")  # spacer
if st.button("✨ Simplify + Translate + Vocabulary", type="primary"):
    if not source_text or not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner(f"Simplifying to {cefr} and translating to {target_lang_ui}..."):
            data = get_scaffolded_content(source_text.strip(), target_lang, cefr)

        if data:
            simplified_placeholder.success(
                data["simplified_text"] if data["simplified_text"] else "(No simplified text returned)"
            )

            translation_placeholder.info(
                data["full_translation"] if data["full_translation"] else "(No translation returned)"
            )

            # Vocab table under the boxes
            st.divider()
            st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

            df = pd.DataFrame(data["vocabulary"])

            expected_cols = ["word", "definition", "translation_word", "translation_definition"]
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = ""

            df = df[expected_cols].rename(columns={
                "word": "Word",
                "definition": "English Definition",
                "translation_word": f"Word Translation ({target_lang_ui})",
                "translation_definition": f"Definition Translation ({target_lang_ui})",
            })

            st.dataframe(df, hide_index=True, use_container_width=True)

        else:
            simplified_placeholder.info("No output yet. Try again.")
            translation_placeholder.info("No output yet. Try again.")
else:
    simplified_placeholder.info("Click the button to generate the simplified text.")
    translation_placeholder.info("Click the button to generate the full translation.")