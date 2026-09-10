import gradio as gr
from ollama import Client

client = Client(host="http://localhost:11434")


def chat_with_ollama(message, history):
    messages = []

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat(
        model="llama3.2",
        messages=messages,
    )

    return response["message"]["content"]


demo = gr.ChatInterface(
    fn=chat_with_ollama,
    title="Gradio + Ollama Chat",
    description="Ask a local Ollama model from a Gradio interface.",
)


if __name__ == "__main__":
    demo.launch()
