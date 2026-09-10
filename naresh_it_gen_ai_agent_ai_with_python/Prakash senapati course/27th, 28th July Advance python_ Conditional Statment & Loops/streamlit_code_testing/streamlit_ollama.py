import streamlit as st
from ollama import Client

# ✅ Create client instance (VERY IMPORTANT)
client = Client(host="http://localhost:11434")

st.set_page_config(
    page_title="Custom LLM model by Ravi Kumar goud  - Ollama",
    layout="centered"
)

st.title("Mr. Ravi Kumar goud  - Ollama App")

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Thinking..."):
            response = client.chat(
                model="gemma3",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            st.success("Response Generated!")
            st.write(response["message"]["content"])