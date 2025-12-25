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
    "Russian": "Russian",
    "Spanish": "Spanish",
    "Thai": "Thai",
    "Turkish": "Turkish",
    "Urdu": "Urdu",
}

LEVEL_OPTIONS = ["Beginner (A2)", "Intermediate (B1)", "Advanced (B2)"]

def extract_cefr(level_label: str) -> str:
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

def parse_protected_terms(raw: str) -> list[str]:
    if not raw:
        return []
    terms = re.split(r"[,\n]+", raw)
    cleaned = []
    for t in terms:
        t = t.strip()
        if t:
            cleaned.append(t)
    seen = set()
    out = []
    for t in cleaned:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

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
# AI function
# -------------------------
def get_scaffolded_content(text: str, language: str, cefr_level: str, protected: list[str]) -> dict | None:
    protected_block = "\n".join([f"- {t}" for t in protected]) if protected else "(none)"

    prompt = f"""
You are an expert EAL teacher and a careful translator.

Protected key terms (do NOT change these exact terms in the simplified text):
{protected_block}

Tasks:
1) Simplify and paraphrase INPUT_TEXT into clear, student-friendly academic English at CEFR {cefr_level}.
   Preserve meaning and keep subject keywords accurate.
   IMPORTANT: Do NOT change the protected key terms (keep the exact spelling).
   Output as SIMPLIFIED_TEXT.

2) Translate the ORIGINAL INPUT_TEXT into {language}. Output as FULL_TRANSLATION.
   FULL_TRANSLATION must be entirely in {language} and natural.

3) Vocabulary:
   Pick exactly 5 difficult academic words from INPUT_TEXT.
   For each word provide:
   - definition: simple English definition
   - translation_word: direct translation of the word into {language}
   - translation_definition: translation of the definition into {language}

4) Comprehension check:
   Create exactly 3 short questions suitable for CEFR {cefr_level} based on SIMPLIFIED_TEXT.
   Each question must have a short, clear answer.

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
  ],
  "questions": [
    {{
      "question": "....?",
      "answer": "...."
    }}
  ]
}}

Rules:
- No markdown.
- No extra keys.
- vocabulary must contain exactly 5 items.
- questions must contain exactly 3 items.
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

        data = json.loads(response.choices[0].message.content)
        if not isinstance(data, dict):
            raise ValueError("Model returned non-object JSON.")

        simplified = safe_get_str(data, "simplified_text")
        full_translation = safe_get_str(data, "full_translation")

        vocab = safe_get_list(data, "vocabulary")
        cleaned_vocab = []
        for item in vocab:
            if isinstance(item, dict):
                cleaned_vocab.append({
                    "word": safe_get_str(item, "word"),
                    "definition": safe_get_str(item, "definition"),
                    "translation_word": safe_get_str(item, "translation_word"),
                    "translation_definition": safe_get_str(item, "translation_definition"),
                })
        cleaned_vocab = cleaned_vocab[:5]
        while len(cleaned_vocab) < 5:
            cleaned_vocab.append({"word": "", "definition": "", "translation_word": "", "translation_definition": ""})

        qs = safe_get_list(data, "questions")
        cleaned_qs = []
        for item in qs:
            if isinstance(item, dict):
                cleaned_qs.append({
                    "question": safe_get_str(item, "question"),
                    "answer": safe_get_str(item, "answer"),
                })
        cleaned_qs = cleaned_qs[:3]
        while len(cleaned_qs) < 3:
            cleaned_qs.append({"question": "", "answer": ""})

        return {
            "simplified_text": simplified,
            "full_translation": full_translation,
            "vocabulary": cleaned_vocab,
            "questions": cleaned_qs
        }

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# -------------------------
# Session state init (so outputs persist when user clicks around)
# -------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None

# -------------------------
# Main UI
# -------------------------
st.title("🎓 Academic Text Helper")
st.markdown(
    "Paste your text to get: simplified English, a full translation of the original text, a vocab table, and quick comprehension questions."
)

# Controls above the input box
col_ctrl1, col_ctrl2 = st.columns([1.2, 1])

with col_ctrl1:
    target_lang_ui = st.selectbox("Translation Language", list(LANGUAGE_MAP.keys()), index=0)
    target_lang = LANGUAGE_MAP[target_lang_ui]

with col_ctrl2:
    level_label = st.selectbox("English Level (Simplification)", LEVEL_OPTIONS, index=0)
    cefr = extract_cefr(level_label)

protected_raw = st.text_input(
    "Protected key terms (optional) - these will NOT be simplified. Separate by commas or new lines.",
    placeholder="e.g. diffusion, osmosis, concentration gradient"
)
protected_terms = parse_protected_terms(protected_raw)

# Side-by-side input and simplified output
col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("📝 Input Text")
    source_text = st.text_area(
        "",
        height=260,
        placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy..."
    )

with col_out:
    st.subheader("📖 Simplified Text (English)")
    if st.session_state["result"] is None:
        st.info("Click **Generate Support** to see the simplified text.")
    else:
        st.success(st.session_state["result"].get("simplified_text") or "(No simplified text returned)")

# Action button
st.markdown("")
if st.button("✨ Generate Support", type="primary"):
    if not source_text or not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner(f"Simplifying to {cefr} and translating to {target_lang_ui}..."):
            data = get_scaffolded_content(
                text=source_text.strip(),
                language=target_lang,
                cefr_level=cefr,
                protected=protected_terms
            )
        if data:
            st.session_state["result"] = data
            st.rerun()

# Outputs
result = st.session_state["result"]

st.subheader(f"🌍 Full Translation of Input Text ({target_lang_ui})")
if result is None:
    st.caption("Generate support first. The translation will appear here.")
else:
    st.info(result.get("full_translation") or "(No translation returned)")

st.divider()
st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

if result is None:
    st.caption("Generate support first. Vocabulary will appear here.")
else:
    df_vocab = pd.DataFrame(result.get("vocabulary", []))
    expected_cols = ["word", "definition", "translation_word", "translation_definition"]
    for c in expected_cols:
        if c not in df_vocab.columns:
            df_vocab[c] = ""
    df_vocab = df_vocab[expected_cols].rename(columns={
        "word": "Word",
        "definition": "English Definition",
        "translation_word": f"Word Translation ({target_lang_ui})",
        "translation_definition": f"Definition Translation ({target_lang_ui})",
    })
    st.dataframe(df_vocab, hide_index=True, use_container_width=True)

st.divider()
st.subheader("✅ Comprehension Check")

if result is None:
    st.caption("Generate support first. Questions will appear here.")
else:
    qs = result.get("questions", [])
    for i, qa in enumerate(qs, start=1):
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q:
            continue
        st.markdown(f"**Q{i}. {q}**")
        with st.expander("Show suggested answer"):
            st.write(a if a else "(No answer returned)")