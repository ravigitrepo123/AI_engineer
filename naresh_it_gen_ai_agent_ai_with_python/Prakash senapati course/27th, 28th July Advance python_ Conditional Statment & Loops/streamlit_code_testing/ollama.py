from ollama import Client

client = Client(host="http://localhost:11434")

response = client.generate(
    model="llama3.2",
    prompt="Hello!"
)

print(response["response"])