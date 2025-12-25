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
    "German": "German",
    "Turkish": "Turkish",
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
st.markdown("Paste your text to get a simplified version and a translated vocabulary list.")

# Single dropdown on top of the input box (as requested)
target_lang_ui = st.selectbox(
    "Translation Language",
    list(LANGUAGE_MAP.keys()),
    index=0
)
target_lang = LANGUAGE_MAP[target_lang_ui]

# Side-by-side input and simplified output
col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("📝 Input Text")
    source_text = st.text_area(
        "",
        height=220,
        placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy..."
    )

with col_out:
    st.subheader("📖 Simplified Text")
    simplified_placeholder = st.empty()

# -------------------------
# AI function
# -------------------------
def get_scaffolded_content(text: str, language: str) -> dict | None:
    """
    Returns:
    - simplified_text (English, simplified)
    - vocabulary: list of {word, definition, translation}
    """
    prompt = f"""
You are an expert EAL teacher.

Tasks:
1) Simplify INPUT_TEXT into clear, student-friendly academic English. Keep the meaning and key physics terms accurate.
2) Select exactly 5 difficult academic words from INPUT_TEXT.
3) For each word provide:
   - definition: simple English definition
   - translation: direct translation into {language}

Return JSON ONLY with EXACTLY this shape:
{{
  "simplified_text": "....",
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
- vocabulary must contain exactly 5 items.

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
        vocab = safe_get_list(data, "vocabulary")

        cleaned_vocab = []
        for item in vocab:
            if not isinstance(item, dict):
                continue
            cleaned_vocab.append({
                "word": safe_get_str(item, "word"),
                "definition": safe_get_str(item, "definition"),
                "translation": safe_get_str(item, "translation"),
            })

        # Hard enforce 5 rows for stable UI
        cleaned_vocab = cleaned_vocab[:5]
        while len(cleaned_vocab) < 5:
            cleaned_vocab.append({"word": "", "definition": "", "translation": ""})

        return {"simplified_text": simplified, "vocabulary": cleaned_vocab}

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# -------------------------
# Action button
# -------------------------
st.markdown("")  # small spacer
if st.button("✨ Simplify & Translate Vocabulary", type="primary"):
    if not source_text or not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner(f"Working in {target_lang_ui}..."):
            data = get_scaffolded_content(source_text.strip(), target_lang)

        if data:
            simplified_placeholder.success(data["simplified_text"] if data["simplified_text"] else "(No simplified text returned)")

            # Vocab table under the input/output box
            st.divider()
            st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

            df = pd.DataFrame(data["vocabulary"])
            for c in ["word", "definition", "translation"]:
                if c not in df.columns:
                    df[c] = ""

            df = df[["word", "definition", "translation"]].rename(columns={
                "word": "Word",
                "definition": "English Definition",
                "translation": f"Translation ({target_lang_ui})"
            })

            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            simplified_placeholder.info("No output yet. Try again.")
else:
    simplified_placeholder.info("Click the button to generate the simplified text.")