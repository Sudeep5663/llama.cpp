import requests
import json
import time

url = 'http://0.0.0.0:8080/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization' : 'Bearer sk-7c9d4b21a2e84f0d8c72f8bb12f6b9c8'
}


def cre_payload(prompt, model="Mistral-7B", temperature=0.1, top_p=0.5, max_tokens=100
                ):
    return {
        'model': model,
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant.',
        },
            {
            'role': 'user',
            'content': prompt
        }],
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
    }


def send_request(promt):
    payload = cre_payload(promt)
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response

prompt = input("Enter your prompt: ") 

start = time.time()
response = send_request(prompt)
end = time.time()

print(response.json()['choices'][0]['message']['content'])
print("Time taken : ",round(end-start,2))
