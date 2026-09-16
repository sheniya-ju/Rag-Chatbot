import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

messages = [
    {
        "role": "system",
        "content": "You are a Python teacher. Explain everything in very simple language."
    }
]

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Chat ended.")
        break

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=2,
        max_tokens=500
    )
   
    answer = response.choices[0].message.content

    print("AI:", answer)

    messages.append({
        "role": "assistant",
        "content": answer
    })