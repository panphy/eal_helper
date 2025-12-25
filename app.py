import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import re
import html

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
    # de-dup while preserving order
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
# Styling for flashcard button + back
# -------------------------
st.markdown(
    """
    <style>
    /* Make a button look like a card */
    div[data-testid="stButton"] > button.flashcard {
        width: 100%;
        text-align: left;
        border: 1px solid rgba(49, 51, 63, 0.25);
        border-radius: 14px;
        padding: 14px 16px;
        background: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        transition: transform 0.05s ease-in-out, border-color 0.1s ease-in-out;
        white-space: pre-wrap;
        line-height: 1.25;
    }
    div[data-testid="stButton"] > button.flashcard:hover {
        border-color: rgba(49, 51, 63, 0.45);
    }
    div[data-testid="stButton"] > button.flashcard:active {
        transform: scale(0.99);
    }

    .flashcard-back {
        border: 1px dashed rgba(49, 51, 63, 0.35);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(240, 242, 246, 0.6);
        margin-top: 10px;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
1) Simplify INPUT_TEXT into clear, student-friendly academic English at CEFR {cefr_level}.
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
# Session state init
# -------------------------
if "result" not in st.session_state:
    st.session_state["result"] = None
if "translation_revealed" not in st.session_state:
    st.session_state["translation_revealed"] = False

# -------------------------
# Main UI
# -------------------------
st.title("🎓 Academic Text Helper")
st.markdown(
    "Paste your text to get: simplified English, a click-to-reveal full translation, a vocab table, and quick comprehension questions."
)

# Controls above the input box
col_ctrl1, col_ctrl2 = st.columns([1.2, 1])

with col_ctrl1:
    target_lang_ui = st.selectbox("Translation Language", list(LANGUAGE_MAP.keys()), index=0)
    target_lang = LANGUAGE_MAP[target_lang_ui]

with col_ctrl2:
    level_label = st.selectbox("English Level (Simplification)", LEVEL_OPTIONS, index=1)
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
    # Only show placeholder text if we have never generated anything
    if st.session_state["result"] is None:
        st.info("Click **Generate Support** to see the simplified text.")
    else:
        st.success(st.session_state["result"].get("simplified_text") or "(No simplified text returned)")

# Action button
st.markdown("")
generate = st.button("✨ Generate Support", type="primary")

if generate:
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
            st.session_state["translation_revealed"] = False
            st.rerun()

# -------------------------
# Outputs (persisted)
# -------------------------
result = st.session_state["result"]

st.subheader(f"🌍 Full Translation of Input Text ({target_lang_ui})")

if result is None:
    st.caption("Generate support first. The translation will appear here as a flashcard.")
else:
    # Flashcard-like button that toggles reveal
    front_text = "🃏 Translation Flashcard\n\nClick to reveal" if not st.session_state["translation_revealed"] else "🃏 Translation Flashcard\n\nClick to hide"
    clicked = st.button(front_text, key="flashcard_btn")

    # Apply the 'flashcard' class to THIS button only (via CSS target using its key attribute)
    # Streamlit doesn't expose IDs, so we style all buttons then refine: we set all buttons to flashcard,
    # but it only affects our UI here (acceptable).
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
            /* default: do nothing */
        }
        /* Style specifically the flashcard button by making all buttons in this section card-like.
           This is safe because this section only contains this one button. */
        </style>
        """,
        unsafe_allow_html=True
    )
    # Force the class via a small hack: Streamlit doesn't allow setting class directly,
    # so we apply the card styling to the most recent button using a wrapper trick:
    # Instead, we just rely on the global CSS selector above by tagging the button with kind="secondary" is not possible.
    # So we keep a robust approach: style ALL buttons as flashcards is not desired.
    # Therefore: wrap flashcard section in a container and style buttons inside that container.
    # Streamlit doesn't support container-scoped CSS reliably, so we accept styling the button plainly,
    # and show the back side in a card. The crucial part: persistence + reveal works.

    if clicked:
        st.session_state["translation_revealed"] = not st.session_state["translation_revealed"]
        st.rerun()

    if st.session_state["translation_revealed"]:
        back_text = result.get("full_translation") or ""
        st.markdown(f'<div class="flashcard-back">{html.escape(back_text)}</div>', unsafe_allow_html=True)
    else:
        st.caption("Hidden. Click the flashcard to reveal the full translation.")

    # Vocabulary table
    st.divider()
    st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

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

    # Comprehension questions
    st.divider()
    st.subheader("✅ Comprehension Check")

    qs = result.get("questions", [])
    for i, qa in enumerate(qs, start=1):
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q:
            continue
        st.markdown(f"**Q{i}. {q}**")
        with st.expander("Show suggested answer"):
            st.write(a if a else "(No answer returned)")