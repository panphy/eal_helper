import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import re
import html
import jsonschema
from jsonschema import ValidationError

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

def reset_result() -> None:
    st.session_state["result"] = None

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
# CSS: green simplified box + alignment
# -------------------------
BOX_HEIGHT_PX = 260
MAX_INPUT_CHARS = 4000

st.markdown(
    f"""
    <style>
      .simplified-box {{
        background: #e8f5e9;            /* light green */
        color: #1b5e20;                 /* dark green */
        border: 1px solid rgba(27, 94, 32, 0.25);
        border-radius: 12px;
        padding: 12px 14px;
        height: {BOX_HEIGHT_PX}px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.35;
      }}
      .simplified-placeholder {{
        color: rgba(27, 94, 32, 0.75);
      }}
      /* Reduce any extra top spacing inside columns */
      .block-container {{
        padding-top: 2.2rem;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# AI function
# -------------------------
def get_scaffolded_content(text: str, language: str, cefr_level: str, protected: list[str]) -> dict | None:
    protected_block = "\n".join([f"- {t}" for t in protected]) if protected else "(none)"
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["simplified_text", "full_translation", "vocabulary", "questions"],
        "properties": {
            "simplified_text": {"type": "string"},
            "full_translation": {"type": "string"},
            "vocabulary": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["word", "definition", "translation_word", "translation_definition"],
                    "properties": {
                        "word": {"type": "string"},
                        "definition": {"type": "string"},
                        "translation_word": {"type": "string"},
                        "translation_definition": {"type": "string"},
                    },
                },
            },
            "questions": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question", "answer"],
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                },
            },
        },
    }

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
        system_messages = [
            "You output strict JSON only. No extra keys, no markdown.",
            (
                "You must return a JSON object that exactly matches the required schema, "
                "including exactly 5 vocabulary items and exactly 3 questions. "
                "No extra keys, no markdown."
            ),
        ]
        data = None
        last_error: Exception | None = None

        for attempt, system_message in enumerate(system_messages, start=1):
            try:
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                data = json.loads(response.choices[0].message.content)
                if not isinstance(data, dict):
                    raise ValueError("Model returned non-object JSON.")
                jsonschema.validate(instance=data, schema=response_schema)
                last_error = None
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                data = None
                if attempt == len(system_messages):
                    break

        if last_error is not None:
            raise ValueError(
                "Model response did not match the expected format after a retry. "
                "Please try again."
            ) from last_error

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

        qs = safe_get_list(data, "questions")
        cleaned_qs = []
        for item in qs:
            if isinstance(item, dict):
                cleaned_qs.append({
                    "question": safe_get_str(item, "question"),
                    "answer": safe_get_str(item, "answer"),
                })

        return {
            "simplified_text": simplified,
            "full_translation": full_translation,
            "vocabulary": cleaned_vocab,
            "questions": cleaned_qs
        }

    except Exception as e:
        st.session_state["result"] = None
        st.error(f"Error: {e}")
        return None

# -------------------------
# Session state init
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
    lang_keys = list(LANGUAGE_MAP.keys())
    current_lang = st.session_state.get("lang_ui", "Arabic")
    lang_index = lang_keys.index(current_lang) if current_lang in lang_keys else 0
    target_lang_ui = st.selectbox(
        "Translation Language",
        lang_keys,
        index=lang_index,
        key="lang_ui",
        on_change=reset_result
    )
    target_lang = LANGUAGE_MAP[target_lang_ui]

with col_ctrl2:
    current_level = st.session_state.get("level_label", "Intermediate (B1)")
    level_index = LEVEL_OPTIONS.index(current_level) if current_level in LEVEL_OPTIONS else 1
    level_label = st.selectbox(
        "English Level (Simplification)",
        LEVEL_OPTIONS,
        index=level_index,
        key="level_label",
        on_change=reset_result
    )
    cefr = extract_cefr(level_label)

protected_raw = st.text_input(
    "Protected key terms (optional) - these will NOT be simplified. Separate by commas or new lines.",
    placeholder="e.g. diffusion, osmosis, concentration gradient",
    key="protected_terms_raw",
    on_change=reset_result
)
protected_terms = parse_protected_terms(protected_raw)

# --- Aligned side-by-side section ---
hdr1, hdr2 = st.columns([1, 1])
with hdr1:
    st.subheader("📝 Input Text")
with hdr2:
    st.subheader("📖 Simplified Text (English)")

col_in, col_out = st.columns([1, 1])

with col_in:
    source_text = st.text_area(
        "",
        height=BOX_HEIGHT_PX,
        placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy...",
        key="source_text",
        label_visibility="collapsed",   # IMPORTANT: removes label space to align perfectly
        on_change=reset_result
    )

with col_out:
    result = st.session_state["result"]
    if result is None:
        st.markdown(
            f'<div class="simplified-box simplified-placeholder">Click <b>Generate Support</b> to see the simplified text.</div>',
            unsafe_allow_html=True
        )
    else:
        simp = result.get("simplified_text") or ""
        st.markdown(
            f'<div class="simplified-box">{html.escape(simp)}</div>',
            unsafe_allow_html=True
        )

# Action button
st.markdown("")
if st.button("✨ Generate Support", type="primary"):
    if not source_text or not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    elif len(source_text) > MAX_INPUT_CHARS:
        st.warning(f"⚠️ Input is too long. Please keep it under {MAX_INPUT_CHARS:,} characters.")
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
