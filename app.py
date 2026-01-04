import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import re
import html
import jsonschema
from jsonschema import ValidationError
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="EAL Learning Companion",
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

ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "is", "in", "that", "for", "on", "with", "as", "are",
    "was", "were", "be", "by", "or", "from", "at", "this", "which", "an", "not",
    "have", "has", "had", "it", "its", "their", "there", "than", "but", "if",
}

def is_likely_english(text: str) -> bool:
    # Heuristic: only flag likely English when Latin-script words dominate and common
    # English stopwords appear, so non-Latin scripts are not penalized.
    if not text:
        return False
    latin_words = re.findall(r"[A-Za-z]+", text)
    all_words = re.findall(r"\w+", text, flags=re.UNICODE)
    if not latin_words or not all_words:
        return False
    english_hits = sum(1 for w in latin_words if w.lower() in ENGLISH_STOPWORDS)
    latin_ratio = len(latin_words) / len(all_words)
    english_ratio = english_hits / len(latin_words)
    return (english_hits >= 2 and english_ratio >= 0.2 and latin_ratio >= 0.6) or (
        english_hits >= 1 and latin_ratio >= 0.85 and len(latin_words) >= 6
    )

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

def clear_input() -> None:
    st.session_state["source_text"] = ""
    reset_result()

def load_preferences() -> dict:
    if not PREFS_PATH.exists():
        return {}
    try:
        data = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

def persist_preferences() -> None:
    prefs = {
        "lang_ui": st.session_state.get("lang_ui"),
        "level_label": st.session_state.get("level_label"),
    }
    try:
        PREFS_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        pass

def handle_selection_change() -> None:
    reset_result()
    persist_preferences()

def render_progress_overlay(percent: int) -> str:
    clamped = max(0, min(percent, 100))
    ring_pos = max(4, min(clamped, 96))
    shade = int(120 + (clamped / 100) * 100)
    shade_green = min(shade + 40, 255)
    bar_style = (
        f"width: {clamped}%;"
        f" background: linear-gradient(90deg, rgba(37, 99, 235, 0.2),"
        f" rgb({shade}, {shade_green}, 255));"
    )
    return f"""
    <div class="ai-overlay">
      <div class="ai-overlay-card">
        <div class="ai-overlay-title">AI is working...</div>
        <div class="ai-overlay-progress">
          <div class="ai-overlay-track">
            <div class="ai-overlay-bar" style="{bar_style}"></div>
          </div>
          <div class="ai-overlay-ring" style="left: {ring_pos}%"></div>
        </div>
        <div class="ai-overlay-percent">{clamped}%</div>
      </div>
    </div>
    """

class TranslationConstraintError(ValueError):
    pass

class ProtectedTermError(ValueError):
    pass

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
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CALLS = 3
SESSION_QUOTA_MAX_CALLS = 20
PREFS_PATH = Path(".eal_helper_prefs.json")

st.markdown(
    f"""
    <style>
      :root {{
        color-scheme: light;
        --color-bg: #f4f6fb;
        --color-surface: #ffffff;
        --color-surface-muted: #eef2f8;
        --color-border: rgba(23, 37, 84, 0.12);
        --color-border-strong: rgba(23, 37, 84, 0.2);
        --color-text: #0b1a33;
        --color-text-muted: rgba(11, 26, 51, 0.62);
        --color-accent: #2f6bff;
        --color-accent-soft: rgba(47, 107, 255, 0.14);
        --success-surface: #e6f6ed;
        --success-text: #0f5132;
        --success-border: rgba(15, 81, 50, 0.28);
        --success-text-muted: rgba(15, 81, 50, 0.72);
        --info-surface: rgba(47, 107, 255, 0.1);
        --info-border: rgba(47, 107, 255, 0.25);
        --info-text: #18349a;
        --warning-surface: rgba(217, 119, 6, 0.1);
        --warning-border: rgba(217, 119, 6, 0.3);
        --warning-text: #8a3d00;
        --overlay-bg: rgba(11, 26, 51, 0.24);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 6px 18px rgba(15, 23, 42, 0.08);
        --shadow-md: 0 10px 26px rgba(15, 23, 42, 0.12);
        --space-1: 4px;
        --space-2: 8px;
        --space-3: 12px;
        --space-4: 16px;
        --space-5: 20px;
        --space-6: 24px;
        --space-7: 32px;
        --font-size-1: 0.85rem;
        --font-size-2: 1rem;
        --font-size-3: 1.25rem;
        --font-size-4: 1.9rem;
      }}
      .stack {{
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
      }}
      .row {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: var(--space-2);
      }}
      .row-between {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-2);
      }}
      .card {{
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        box-shadow: var(--shadow-sm);
      }}
      .card:hover {{
        border-color: var(--color-border-strong);
        box-shadow: var(--shadow-md);
      }}
      .card-header {{
        margin-bottom: var(--space-2);
      }}
      .card-body {{
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
      }}
      .title {{
        margin: 0;
        font-size: var(--font-size-4);
        font-weight: 700;
        color: var(--color-text);
      }}
      .subtitle {{
        margin: 0;
        font-size: var(--font-size-2);
        color: var(--color-text-muted);
      }}
      .text-muted {{
        color: var(--color-text-muted);
      }}
      .text-sm {{
        font-size: var(--font-size-1);
      }}
      .pill {{
        display: inline-flex;
        align-items: center;
        gap: var(--space-1);
        padding: var(--space-1) var(--space-2);
        border-radius: 999px;
        background: var(--color-accent-soft);
        color: var(--color-accent);
        font-size: var(--font-size-1);
        font-weight: 600;
        border: 1px solid transparent;
      }}
      .pill:hover {{
        border-color: var(--color-accent);
      }}
      .box {{
        background: var(--color-surface-muted);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: var(--space-3);
      }}
      .box-success {{
        background: var(--success-surface);
        color: var(--success-text);
        border-color: var(--success-border);
      }}
      .box-scroll {{
        height: {BOX_HEIGHT_PX}px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.4;
      }}
      .text-success-muted {{
        color: var(--success-text-muted);
      }}
      .app-hero {{
        background: linear-gradient(120deg, #f0f4ff, #ecf8f1);
        margin-bottom: var(--space-5);
      }}
      .stApp {{
        background-color: var(--color-bg);
        color: var(--color-text);
      }}
      .stMarkdown, .stCaption, .stTextInput label, .stSelectbox label, .stTextArea label, .stCheckbox label {{
        color: var(--color-text);
      }}
      .stCaption {{
        color: var(--color-text-muted);
      }}
      .stTextArea textarea,
      .stTextInput input,
      .stSelectbox div[data-baseweb="select"] > div {{
        background: var(--color-surface);
        color: var(--color-text);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
      }}
      .stTextArea textarea::placeholder,
      .stTextInput input::placeholder {{
        color: var(--color-text-muted);
      }}
      .stSelectbox div[data-baseweb="select"] span {{
        color: var(--color-text);
      }}
      div[data-testid="stAlert"] {{
        background: var(--info-surface);
        border: 1px solid var(--info-border);
        color: var(--info-text);
        border-radius: var(--radius-md);
      }}
      div[data-testid="stAlert"] svg {{
        color: var(--info-text);
      }}
      div[data-testid="stAlert"][data-alert-type="warning"] {{
        background: var(--warning-surface);
        border-color: var(--warning-border);
        color: var(--warning-text);
      }}
      div[data-testid="stAlert"][data-alert-type="warning"] svg {{
        color: var(--warning-text);
      }}
      div[data-testid="stDataFrame"] {{
        color: var(--color-text);
        border-radius: var(--radius-md);
        overflow: hidden;
      }}
      div[data-testid="stDataFrame"] thead,
      div[data-testid="stDataFrame"] tbody {{
        color: var(--color-text);
      }}
      div[data-testid="stExpander"] {{
        border-radius: var(--radius-md);
        overflow: hidden;
      }}
      .stButton > button {{
        border-radius: var(--radius-md);
      }}
      .ai-overlay {{
        position: fixed;
        inset: 0;
        background: var(--overlay-bg);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        pointer-events: all;
      }}
      .ai-overlay-card {{
        width: min(360px, 80vw);
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-5);
        box-shadow: var(--shadow-md);
        text-align: center;
      }}
      .ai-overlay-title {{
        font-size: var(--font-size-2);
        font-weight: 600;
        color: var(--color-text);
        margin-bottom: var(--space-3);
      }}
      .ai-overlay-progress {{
        width: 100%;
        height: 14px;
        overflow: visible;
        position: relative;
      }}
      .ai-overlay-track {{
        width: 100%;
        height: 10px;
        background: var(--color-surface-muted);
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--color-border);
        position: absolute;
        top: 50%;
        left: 0;
        transform: translateY(-50%);
      }}
      .ai-overlay-percent {{
        margin-top: var(--space-2);
        font-size: var(--font-size-1);
        color: var(--color-text-muted);
        font-weight: 600;
        letter-spacing: 0.02em;
      }}
      .ai-overlay-bar {{
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.2), var(--color-accent));
        border-radius: 999px;
        transition: width 180ms ease-out, background 300ms ease-out;
      }}
      .ai-overlay-ring {{
        position: absolute;
        top: 50%;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid rgba(47, 107, 255, 0.35);
        border-top-color: var(--color-accent);
        transform: translate(-50%, -50%);
        box-shadow: 0 0 6px rgba(47, 107, 255, 0.35);
        animation: ring-spin 0.9s linear infinite, ring-pulse 1.6s ease-in-out infinite;
        background: rgba(255, 255, 255, 0.7);
      }}
      @keyframes ring-spin {{
        from {{ transform: translate(-50%, -50%) rotate(0deg); }}
        to {{ transform: translate(-50%, -50%) rotate(360deg); }}
      }}
      @keyframes ring-pulse {{
        0%, 100% {{ box-shadow: 0 0 6px rgba(47, 107, 255, 0.35); }}
        50% {{ box-shadow: 0 0 10px rgba(47, 107, 255, 0.6); }}
      }}
      #MainMenu {{ visibility: hidden; }}
      button[data-testid="stMainMenu"] {{ visibility: hidden; }}
      button[title="View settings"] {{ visibility: hidden; }}
      button[title="Settings"] {{ visibility: hidden; }}
      /* Reduce any extra top spacing inside columns */
      .block-container {{
        padding-top: calc(var(--space-7) + 24px);
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
            (
                "Return ONLY valid JSON. Do not include any extra text outside the JSON "
                "object. The JSON must match the required schema exactly."
            ),
        ]
        data = None
        last_error: Exception | None = None
        protected_retry_used = False

        for attempt, system_message in enumerate(system_messages, start=1):
            try:
                user_prompt = prompt
                if attempt == len(system_messages):
                    user_prompt = (
                        "Return ONLY valid JSON; do not include text outside JSON.\n\n"
                        f"{prompt}"
                    )
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                data = json.loads(response.choices[0].message.content)
                if not isinstance(data, dict):
                    raise ValueError("Model returned non-object JSON.")
                jsonschema.validate(instance=data, schema=response_schema)
                simplified = safe_get_str(data, "simplified_text")
                full_translation = safe_get_str(data, "full_translation")
                vocab = safe_get_list(data, "vocabulary")
                translation_definitions = []
                for item in vocab:
                    if isinstance(item, dict):
                        translation_definitions.append(
                            safe_get_str(item, "translation_definition")
                        )
                missing_terms = [term for term in protected if term not in simplified]
                if missing_terms:
                    raise ProtectedTermError(
                        "Protected terms missing from simplified_text: "
                        + ", ".join(missing_terms)
                    )
                violation_fields = []
                if is_likely_english(full_translation):
                    violation_fields.append("full_translation")
                for index, definition in enumerate(translation_definitions, start=1):
                    if is_likely_english(definition):
                        violation_fields.append(
                            f"vocabulary[{index}].translation_definition"
                        )
                if violation_fields:
                    raise TranslationConstraintError(
                        "Translation appears to contain English in: "
                        + ", ".join(violation_fields)
                    )
                last_error = None
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                data = None
                if isinstance(exc, ProtectedTermError):
                    if protected_retry_used or attempt == len(system_messages):
                        break
                    protected_retry_used = True
                    time.sleep(0.5)
                    continue
                if attempt == len(system_messages):
                    break
                time.sleep(0.5)

        if last_error is not None:
            if isinstance(last_error, ProtectedTermError):
                raise ValueError(
                    "Protected terms must appear verbatim in the simplified text. "
                    "Missing terms: " + str(last_error).replace(
                        "Protected terms missing from simplified_text: ", ""
                    )
                ) from last_error
            if isinstance(last_error, TranslationConstraintError):
                raise ValueError(
                    "Translation constraint failure: please ensure the full translation "
                    f"and vocabulary definitions are entirely in {language} with no "
                    "English words."
                ) from last_error
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
if "call_times" not in st.session_state:
    st.session_state["call_times"] = []
if "call_count" not in st.session_state:
    st.session_state["call_count"] = 0
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
saved_prefs = load_preferences()
default_lang = saved_prefs.get("lang_ui") if isinstance(saved_prefs, dict) else None
if not isinstance(default_lang, str) or default_lang not in LANGUAGE_MAP:
    default_lang = "Arabic"
default_level = saved_prefs.get("level_label") if isinstance(saved_prefs, dict) else None
if not isinstance(default_level, str) or default_level not in LEVEL_OPTIONS:
    default_level = "Intermediate (B1)"

# -------------------------
# Main UI
# -------------------------
st.markdown(
    """
    <div class="card app-hero stack">
      <div class="card-header">
        <h1 class="title">🎓 EAL Learning Companion</h1>
      </div>
      <div class="card-body">
        <p class="subtitle">Paste a passage to get simplified English, a faithful translation, vocabulary support, and comprehension checks in one place.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Controls above the input box
col_ctrl1, col_ctrl2 = st.columns([1.1, 1])

with col_ctrl1:
    lang_keys = list(LANGUAGE_MAP.keys())
    current_lang = st.session_state.get("lang_ui", default_lang)
    lang_index = lang_keys.index(current_lang) if current_lang in lang_keys else 0
    target_lang_ui = st.selectbox(
        "Translation Language",
        lang_keys,
        index=lang_index,
        key="lang_ui",
        on_change=handle_selection_change
    )
    target_lang = LANGUAGE_MAP[target_lang_ui]

with col_ctrl2:
    current_level = st.session_state.get("level_label", default_level)
    level_index = LEVEL_OPTIONS.index(current_level) if current_level in LEVEL_OPTIONS else 1
    level_label = st.selectbox(
        "Your English Level",
        LEVEL_OPTIONS,
        index=level_index,
        key="level_label",
        on_change=handle_selection_change
    )
    cefr = extract_cefr(level_label)

protected_raw = st.text_input(
    "Protected key terms (optional) - these will NOT be simplified. Separate by commas or new lines.",
    placeholder="e.g. diffusion, osmosis, concentration gradient",
    key="protected_terms_raw",
    on_change=reset_result
)
protected_terms = parse_protected_terms(protected_raw)

# Quick status row
status_left, status_right = st.columns([1, 1])
with status_left:
    st.markdown(
        f'<div class="row">'
        f'<span class="pill">CEFR: {cefr}</span>'
        f'<span class="pill">Protected terms: {len(protected_terms)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with status_right:
    st.caption(f"Max input length: {MAX_INPUT_CHARS:,} characters")

# --- Aligned side-by-side section ---
hdr1, hdr2 = st.columns([1, 1])
with hdr1:
    st.subheader("📝 Input Text")
with hdr2:
    st.subheader("📖 Simplified Text (English)")

col_in, col_out = st.columns([1, 1])

with col_in:
    source_text = st.text_area(
        "Input text",
        height=BOX_HEIGHT_PX,
        placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy...",
        key="source_text",
        label_visibility="collapsed",   # IMPORTANT: removes label space to align perfectly
        on_change=reset_result
    )
    current_len = len(source_text or "")
    st.caption(f"{current_len:,} / {MAX_INPUT_CHARS:,} characters")

with col_out:
    result = st.session_state["result"]
    if result is None:
        st.markdown(
            (
                f'<div class="box box-success box-scroll text-success-muted">'
                f'Click <b>Generate Support</b> to see the simplified text.'
                f'</div>'
            ),
            unsafe_allow_html=True
        )
    else:
        simp = result.get("simplified_text") or ""
        st.markdown(
            f'<div class="box box-success box-scroll">{html.escape(simp)}</div>',
            unsafe_allow_html=True
        )

# Action button
st.markdown("")
action_col, clear_col = st.columns([1, 1])
with action_col:
    generate_clicked = st.button("✨ Generate Support", type="primary")
with clear_col:
    st.button("🧹 Clear", type="secondary", on_click=clear_input)

if generate_clicked:
    if not source_text or not source_text.strip():
        st.warning("⚠️ Please paste some text first.")
    elif len(source_text) > MAX_INPUT_CHARS:
        st.warning(f"⚠️ Input is too long. Please keep it under {MAX_INPUT_CHARS:,} characters.")
    else:
        now = time.time()
        call_times = [
            t for t in st.session_state["call_times"]
            if now - t < RATE_LIMIT_WINDOW_SECONDS
        ]
        st.session_state["call_times"] = call_times

        if st.session_state["call_count"] >= SESSION_QUOTA_MAX_CALLS:
            st.warning(
                "⚠️ Session quota reached. Please refresh later or start a new session."
            )
        elif len(call_times) >= RATE_LIMIT_MAX_CALLS:
            wait_seconds = int(
                RATE_LIMIT_WINDOW_SECONDS - (now - min(call_times))
            )
            st.warning(
                f"⚠️ Too many requests. Please wait {wait_seconds} seconds and try again."
            )
        else:
            st.session_state["is_processing"] = True
            overlay_placeholder = st.empty()
            overlay_placeholder.markdown(
                render_progress_overlay(0),
                unsafe_allow_html=True,
            )
            st.session_state["call_times"] = call_times + [now]
            st.session_state["call_count"] += 1
            data = None
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        get_scaffolded_content,
                        text=source_text.strip(),
                        language=target_lang,
                        cefr_level=cefr,
                        protected=protected_terms,
                    )
                    current = 0
                    while current < 95 and not future.done():
                        remaining = 95 - current
                        current += max(1, round(remaining / 24))
                        overlay_placeholder.markdown(
                            render_progress_overlay(current),
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.28)
                    while not future.done():
                        overlay_placeholder.markdown(
                            render_progress_overlay(95),
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.35)
                    data = future.result()
                    overlay_placeholder.markdown(
                        render_progress_overlay(100),
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.1)
            finally:
                st.session_state["is_processing"] = False
                overlay_placeholder.empty()
            if data:
                st.session_state["result"] = data
                st.rerun()

# Outputs
result = st.session_state["result"]

st.divider()
tabs = st.tabs(
    [
        f"🌍 Translation ({target_lang_ui})",
        f"🔑 Vocabulary ({target_lang_ui})",
        "✅ Comprehension Check",
    ]
)

with tabs[0]:
    if result is None:
        st.caption("Generate support first. The translation will appear here.")
    else:
        st.info(result.get("full_translation") or "(No translation returned)")

with tabs[1]:
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
        st.dataframe(df_vocab, hide_index=True, width="stretch")

with tabs[2]:
    if result is None:
        st.caption("Generate support first. Questions will appear here.")
    else:
        qs = result.get("questions", [])
        counter = 0
        for qa in qs:
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            if not q:
                continue
            counter += 1
            st.markdown(f"**Q{counter}. {q}**")
            with st.expander("Show suggested answer"):
                st.write(a if a else "(No answer returned)")
