import streamlit as st
from openai import OpenAI
import pandas as pd
import json

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="EAL Scaffolder (GPT-5)",
    page_icon="🚀",
    layout="wide"
)

# --- SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Check for secret key first (for deployment), otherwise ask user
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ Teacher's API Key Loaded")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here.")
    
    st.divider()
    
    # Customization Options
    target_lang = st.selectbox(
        "Student's Home Language", 
        ["Spanish", "Polish", "Arabic", "Urdu", "Chinese", "French", "None (English Only)"]
    )
    
    difficulty = st.select_slider(
        "Simplification Level",
        options=["Beginner (A2)", "Intermediate (B1)", "Advanced (B2)"],
        value="Intermediate (B1)"
    )

# --- MAIN PAGE ---
st.title("🚀 Academic Text Scaffolder")
st.caption("Powered by OpenAI GPT-5 Mini")

st.markdown("""
**Instructions:** Paste a complex academic text below. 
The AI will simplify the language and extract key vocabulary using the latest GPT-5 reasoning.
""")

# Input Area
source_text = st.text_area("Paste your academic text here:", height=200)

# --- THE AI BRAIN ---
def get_scaffolded_content(text, language, level):
    """
    Sends the text to OpenAI GPT-5 Mini and requests a JSON response.
    """
    if not api_key:
        st.error("Please add an API Key in the sidebar.")
        return None
        
    client = OpenAI(api_key=api_key)
    
    # GPT-5 Prompting Strategy:
    # GPT-5 handles complex instruction following much better than GPT-4.
    # We can be very specific about the 'tone' and 'nuance' of the definition.
    prompt = f"""
    You are an expert EAL specialist. Analyze the text below.
    
    1. **Simplify:** Rewrite the text for CEFR Level {level}. Keep the original meaning but use simpler sentence structures.
    2. **Extract:** Identify the 5 most critical academic words (Tier 2/3) that block comprehension.
    3. **Define:** Provide a simple English definition AND a translation in {language}.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "summary": "The simplified text...",
        "vocabulary": [
            {{"word": "example", "definition": "simple definition", "translation": "translated word"}}
        ]
    }}
    
    Text to analyze:
    "{text}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",  # UPDATED TO LATEST MODEL
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
if st.button("✨ Scaffold Text", type="primary"):
    if not source_text:
        st.warning("⚠️ Please paste some text to analyze.")
    else:
        with st.spinner("GPT-5 is analyzing..."):
            data = get_scaffolded_content(source_text, target_lang, difficulty)
            
            if data:
                # Layout: Two columns for side-by-side comparison
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📖 Simplified Version")
                    st.success(data["summary"])
                    
                with col2:
                    st.subheader(f"🔑 Key Vocabulary ({target_lang})")
                    df = pd.DataFrame(data["vocabulary"])
                    st.dataframe(
                        df, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "word": "Word",
                            "definition": "Definition",
                            "translation": f"Translation"
                        }
                    )
