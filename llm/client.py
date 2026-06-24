import ollama

MODEL = "llama3.1:8b"

def chat(messages : list[dict]) -> str:
    response = ollama.chat(
        model = MODEL,
        messages = messages
    )

    return response["message"]["content"]

