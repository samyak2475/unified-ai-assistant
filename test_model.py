import ollama

response = ollama.chat(
    model='phi3:mini',
    messages=[{'role': 'user', 'content': 'Hello, who are you? Give a short answer.'}]
)

print(response['message']['content'])
