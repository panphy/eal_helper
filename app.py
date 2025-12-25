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
# Flashcard HTML (click to reveal)
# -------------------------
def render_flashcard(title: str, back_text: str, key: str):
    """
    Uses Streamlit session_state to reveal/hide translation.
    The card itself is clickable via a button styled as a card.
    """
    if key not in st.session_state:
        st.session_state[key] = False  # False = hidden

    # Styled "card button"
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button.flashcard-btn {
            width: 100%;
            text-align: left;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 14px;
            padding: 14px 16px;
            background: white;
            transition: transform 0.05s ease-in-out;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        div[data-testid="stButton"] > button.flashcard-btn:hover {
            border-color: rgba(49, 51, 63, 0.35);
        }
        div[data-testid="stButton"] > button.flashcard-btn:active {
            transform: scale(0.99);
        }
        .flashcard-title {
            font-weight: 700;
            font-size: 1.0rem;
            margin-bottom: 6px;
        }
        .flashcard-hint {
            opacity: 0.75;
            font-size: 0.95rem;
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

    btn_label = f"{title}\n\nClick to {'hide' if st.session_state[key] else 'reveal'}"
    # We use a normal button but style it as a card.
    clicked = st.button(btn_label, key=f"{key}_btn", type="secondary")
    # Patch the button class via small JS-free hack: Streamlit adds the class name from the label?
    # Instead, we rely on :has() not available. We'll set button style by injecting CSS to target the last button.
    # Workaround: use st.button and then set CSS to all buttons; acceptable for this app.
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="secondary"]{
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if clicked:
        st.session_state[key] = not st.session_state[key]

    if st.session_state[key]:
        safe_text = html.escape(back_text or "")
        st.markdown(f'<div class="flashcard-back">{safe_text}</div>', unsafe_allow_html=True)
    else:
        st.caption("Hidden. Click the card to reveal the full translation.")

# -------------------------
# Main UI
# -------------------------
st.title("🎓 Academic Text Helper")
st.markdown(
    "Paste your text to get: simplified English, a hidden (click-to-reveal) full translation, a vocab table, and quick comprehension questions."
)

# Controls above the input box
col_ctrl1, col_ctrl2 = st.columns([1.2, 1])

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
    simplified_placeholder = st.empty()

# Translation flashcard section (under the side-by-side)
st.subheader(f"🌍 Full Translation of Input Text ({target_lang_ui})")
translation_container = st.container()

# -------------------------
# AI function
# -------------------------
def get_scaffolded_content(
    text: str,
    language: str,
    cefr_level: str,
    protected: list[str]
) -> dict | None:
    """
    Returns:
    - simplified_text (English, simplified to CEFR)
    - full_translation (translation of ORIGINAL input text)
    - vocabulary: list of {word, definition, translation_word, translation_definition}
    - questions: list of {question, answer}
    """
    protected_block = ""
    if protected:
        protected_block = "\n".join([f"- {t}" for t in protected])

    prompt = f"""
You are an expert EAL teacher and a careful translator.

Protected key terms (do NOT change these exact terms in the simplified text):
{protected_block if protected_block else "(none)"}

Tasks:
1) Simplify INPUT_TEXT into clear, student-friendly academic English at CEFR {cefr_level}.
   Preserve meaning and keep subject keywords accurate.
   IMPORTANT: Do NOT change the protected key terms (keep the exact spelling).
   Output as SIMPLIFIED_TEXT.

2) Translate the ORIGINAL INPUT_TEXT into {language}. Output as FULL_TRANSLATION.
   FULL_TRANSLATION must be entirely in {language} and natural.

3) Vocabulary:
   Pick exactly 5 difficult academic words from INPUT_TEXT (not random).
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
        cleaned_vocab = cleaned_vocab[:5]
        while len(cleaned_vocab) < 5:
            cleaned_vocab.append({"word": "", "definition": "", "translation_word": "", "translation_definition": ""})

        qs = safe_get_list(data, "questions")
        cleaned_qs = []
        for item in qs:
            if not isinstance(item, dict):
                continue
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
# Action button
# -------------------------
st.markdown("")
if st.button("✨ Generate Support", type="primary"):
    if not source_text or not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        # Reset flashcard hidden state for new content
        st.session_state["translation_revealed"] = False

        with st.spinner(f"Simplifying to {cefr} and translating to {target_lang_ui}..."):
            data = get_scaffolded_content(
                text=source_text.strip(),
                language=target_lang,
                cefr_level=cefr,
                protected=protected_terms
            )

        if data:
            simplified_text = data["simplified_text"]
            full_translation = data["full_translation"]

            simplified_placeholder.success(simplified_text if simplified_text else "(No simplified text returned)")

            with translation_container:
                render_flashcard(
                    title="🃏 Translation Flashcard",
                    back_text=full_translation or "",
                    key="translation_revealed"
                )

            # Vocabulary table
            st.divider()
            st.subheader(f"🔑 Vocabulary ({target_lang_ui})")

            df_vocab = pd.DataFrame(data["vocabulary"])
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

            for i, qa in enumerate(data.get("questions", []), start=1):
                q = (qa.get("question") or "").strip()
                a = (qa.get("answer") or "").strip()
                if not q:
                    continue
                st.markdown(f"**Q{i}. {q}**")
                with st.expander("Show suggested answer"):
                    st.write(a if a else "(No answer returned)")
        else:
            simplified_placeholder.info("No output yet. Try again.")
            with translation_container:
                st.caption("No translation yet.")
else:
    simplified_placeholder.info("Click the button to generate the simplified text.")
    with translation_container:
        st.caption("Translation is hidden until you generate support, then click the flashcard to reveal.")