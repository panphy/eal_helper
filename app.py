import streamlit as st
from openai import OpenAI
import pandas as pd
import json

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="EAL Scaffolder",
    page_icon="🎓",
    layout="wide"
)

# --- SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. API Key Check (Silent)
    # The app will simply fail gracefully if you haven't set this up in Streamlit Cloud yet.
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("🚨 Admin Error: OpenAI API Key not found in secrets.")
        st.stop()
    
    st.divider()
    
    # 2. Student Options
    target_lang = st.selectbox(
        "Translation Language", 
        ["Spanish", "Polish", "Arabic", "Urdu", "Chinese", "French", "Japanese", "Portuguese"]
    )
    
    difficulty = st.select_slider(
        "Simplification Level",
        options=["Beginner (A2)", "Intermediate (B1)", "Advanced (B2)"],
        value="Intermediate (B1)"
    )

# --- MAIN PAGE ---
st.title("🎓 Academic Text Helper")
st.markdown("Paste your difficult text below to get a simplified version and a translated vocabulary list.")

# Input Area
source_text = st.text_area("Paste text here:", height=200, placeholder="Example: Photosynthesis is the process used by plants to convert light energy into chemical energy...")

# --- THE AI BRAIN ---
def get_scaffolded_content(text, language, level):
    client = OpenAI(api_key=api_key)
    
    # PROMPT ENGINEERING
    # We strictly enforce TWO distinct fields for the vocabulary to ensure translation happens.
    prompt = f"""
    You are an expert EAL teacher. Analyze the provided text.
    
    1. **Simplify:** Rewrite the text for CEFR Level {level}. Keep the meaning but use simpler grammar.
    2. **Extract:** Identify the 5 most difficult academic words.
    3. **Translate:** For each word, provide:
       - A simple English definition.
       - A direct translation into {language}.
    
    Return JSON ONLY:
    {{
        "summary": "The simplified text...",
        "vocabulary": [
            {{
                "word": "English Word", 
                "definition": "Simple English Definition", 
                "translation": "Word in {language}"
            }}
        ]
    }}
    
    Input Text:
    "{text}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- ACTION BUTTON ---
if st.button("✨ Simplify & Translate", type="primary"):
    if not source_text:
        st.warning("⚠️ Please paste some text first.")
    else:
        with st.spinner(f"Translating to {target_lang} with GPT-5 Mini..."):
            data = get_scaffolded_content(source_text, target_lang, difficulty)
            
            if data:
                # Layout: Side-by-Side
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📖 Simplified Text")
                    st.success(data["summary"])
                    
                with col2:
                    st.subheader(f"🔑 Vocabulary ({target_lang})")
                    df = pd.DataFrame(data["vocabulary"])
                    
                    # Force column order and naming for clarity
                    st.dataframe(
                        df, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "word": "Word",
                            "definition": "English Definition",
                            "translation": f"Translation ({target_lang})"
                        }
                    )
