import streamlit as st
from ollama import Client

st.set_page_config(page_title="Ollama Chat", page_icon="🤖")
st.title("Ollama Chat with Streamlit")

prompt = st.text_area("Enter your prompt", "Write a short greeting in Python")

if st.button("Generate"):
    client = Client(host="http://localhost:11434")
    response = client.generate(
        model="llama3.2",
        prompt=prompt,
    )
    st.success("Response")
    st.write(response["response"])
