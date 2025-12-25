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

def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter (good enough for classroom use).
    Keeps punctuation.
    """
    text = (text or "").strip()
    if not text:
        return []
    # Split on . ! ? followed by whitespace, keep punctuation in previous sentence
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def parse_protected_terms(raw: str) -> list[str]:
    """
    Accepts comma/newline separated terms.
    """
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
# Main UI
# -------------------------
st.title("🎓 Academic Text Helper")
st.markdown(
    "Paste your text to get: simplified English, a full translation of the original text, a vocab table, and quick comprehension questions."
)

# Controls above the input box
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1, 1])

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

with col_ctrl3:
    model_name = st.selectbox(
        "Model",
        ["gpt-5-nano", "gpt-5-mini"],
        index=1
    )

# Optional: protected key terms
protected_raw = st.text_input(
    "Protected key terms (optional) - these will NOT be simplified. Separate by commas or new lines.",
    placeholder="e.g. diffusion, osmosis, concentration gradient"
)
protected_terms = parse_protected_terms(protected_raw)

# Optional: sentence-by-sentence display
sentence_mode = st.toggle("Sentence-by-sentence view", value=False)

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

# Full translation area
st.subheader(f"🌍 Full Translation of Input Text ({target_lang_ui})")
translation_placeholder = st.empty()

# -------------------------
# AI function
# -------------------------
def get_scaffolded_content(
    text: str,
    language: str,
    cefr_level: str,
    protected: list[str],
    model: str
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

You will receive INPUT_TEXT.

Protected key terms (do NOT change these exact terms in the simplified text):
{protected_block if protected_block else "(none)"}

Tasks:
1) Simplify INPUT_TEXT into clear, student-friendly academic English at CEFR {cefr_level}.
   Preserve meaning and keep subject keywords accurate.
   IMPORTANT: Do NOT change the protected key terms (keep the exact spelling).
   Output as SIMPLIFIED_TEXT.

2) Translate the ORIGINAL INPUT_TEXT into {language}. Output as FULL_TRANSLATION.
   FULL_TRANSLATION must be entirely in {language} and natural (not word-for-word).

3) Vocabulary:
   Pick exactly 5 difficult academic words from INPUT_TEXT (not random).
   For each word provide:
   - definition: simple English definition
   - translation_word: direct translation of the word into {language}
   - translation_definition: translation of the definition into {language}

4) Comprehension check:
   Create exactly 3 short questions suitable for CEFR {cefr_level} based on SIMPLIFIED_TEXT.
   Each question must have a short, clear answer.
   Output as QUESTIONS.

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
- Keep questions short and direct (good for EAL).

INPUT_TEXT:
{text}
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
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
        with st.spinner(f"Simplifying to {cefr} and translating to {target_lang_ui}..."):
            data = get_scaffolded_content(
                text=source_text.strip(),
                language=target_lang,
                cefr_level=cefr,
                protected=protected_terms,
                model=model_name
            )

        if data:
            simplified_text = data["simplified_text"]
            full_translation = data["full_translation"]

            # Update main outputs
            simplified_placeholder.success(simplified_text if simplified_text else "(No simplified text returned)")
            translation_placeholder.info(full_translation if full_translation else "(No translation returned)")

            # Sentence-by-sentence view
            if sentence_mode:
                st.divider()
                st.subheader("🧩 Sentence-by-sentence (Original → Simplified)")
                orig_sents = split_sentences(source_text.strip())
                simp_sents = split_sentences(simplified_text)

                # Align lengths (simple alignment)
                n = max(len(orig_sents), len(simp_sents))
                while len(orig_sents) < n:
                    orig_sents.append("")
                while len(simp_sents) < n:
                    simp_sents.append("")

                rows = []
                for i in range(n):
                    rows.append({
                        "Original (English)": orig_sents[i],
                        "Simplified (English)": simp_sents[i],
                    })

                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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

            qs = data.get("questions", [])
            for i, qa in enumerate(qs, start=1):
                q = qa.get("question", "").strip()
                a = qa.get("answer", "").strip()
                if not q:
                    continue
                st.markdown(f"**Q{i}. {q}**")
                with st.expander("Show suggested answer"):
                    st.write(a if a else "(No answer returned)")

        else:
            simplified_placeholder.info("No output yet. Try again.")
            translation_placeholder.info("No output yet. Try again.")
else:
    simplified_placeholder.info("Click the button to generate the simplified text.")
    translation_placeholder.info("Click the button to generate the full translation.")