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

# --- HELPERS ---
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
    """
    Converts "Intermediate (B1)" -> "B1" (fallback to label if not found).
    """
    m = re.search(r"\((A1|A2|B1|B2|C1|C2)\)", level_label)
    return m.group(1) if m else level_label

@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# --- SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.header("⚙️ Settings")

    # API Key check (Silent)
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("🚨 Admin Error: OpenAI API Key not found in secrets.")
        st.stop()

    st.divider()

    target_lang_ui = st.selectbox(
        "Translation Language",
        list(LANGUAGE_MAP.keys())
    )
    target_lang = LANGUAGE_MAP[target_lang_ui]

    difficulty_label = st.select_slider(
        "Simplification Level",
        options=["Beginner (A2)", "Intermediate (B1)", "Advanced (B2)"],
        value="Intermediate (B1)"
    )
    cefr = extract_cefr(difficulty_label)

# --- MAIN PAGE ---
st.title("🎓 Academic Text Helper")
st.markdown("Paste your difficult text below to get a simplified version, a full translation, and a translated vocabulary list.")

source_text = st.text_area(
    "Paste text here:",
    height=200,
    placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy..."
)

# --- THE AI BRAIN ---
def get_scaffolded_content(text: str, language: str, cefr_level: str):
    client = get_client(api_key)

    # IMPORTANT FIX:
    # - We now request a FULL translated text output (translated_text)
    # - We keep vocab translations too
    prompt = f"""
You are an expert EAL teacher.

Task:
1) Simplify the INPUT TEXT into clear academic English at CEFR {cefr_level}. Preserve meaning.
2) Translate the simplified text into {language}. The translation must be natural and accurate.
3) Select the 5 most difficult academic words from the INPUT TEXT (not random).
   For each word provide:
   - A simple English definition
   - A direct {language} translation

Return JSON ONLY with EXACTLY this shape:
{{
  "simplified_english": "....",
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
- "translated_text" must be entirely in {language} (do not include English).
- No romanization unless the language normally uses Latin script.
- Keep "vocabulary" as exactly 5 items.

INPUT TEXT:
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

        content = response.choices[0].message.content
        data = json.loads(content)

        # Basic validation / guardrails (prevents KeyError crashes)
        if not isinstance(data, dict):
            raise ValueError("Model returned non-object JSON.")
        for k in ["simplified_english", "translated_text", "vocabulary"]:
            if k not in data:
                raise ValueError(f"Missing key in model output: {k}")
        if not isinstance(data["vocabulary"], list):
            raise ValueError("vocabulary must be a list.")

        return data

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- ACTION BUTTON ---
if st.button("✨ Simplify & Translate", type="primary"):
    if not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner(f"Simplifying (CEFR {cefr}) and translating to {target_lang_ui}..."):
            data = get_scaffolded_content(source_text, target_lang, cefr)

        if data:
            # Layout: 2 columns for text, then vocab table
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📖 Simplified English")
                st.success(data["simplified_english"])

            with col2:
                st.subheader(f"🌍 Translation ({target_lang_ui})")
                st.info(data["translated_text"])

            st.divider()
            st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

            df = pd.DataFrame(data["vocabulary"])
            if not df.empty:
                # Ensure consistent column order even if model changes order
                for col in ["word", "definition", "translation"]:
                    if col not in df.columns:
                        df[col] = ""
                df = df[["word", "definition", "translation"]]

            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "word": "Word",
                    "definition": "English Definition",
                    "translation": f"Translation ({target_lang_ui})"
                }
            )