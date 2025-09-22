from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from queue import Queue
from threading import Thread
import httpx
import time
import json
import tiktoken

app = FastAPI()

request_queue = Queue()
concurrent_requests = 3
semaphore = asyncio.Semaphore(concurrent_requests)

tokenizer = tiktoken.get_encoding("cl100k_base")

url = "http://0.0.0.0:8080/v1/chat/completions"


headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-7c9d4b21a2e84f0d8c72f8bb12f6b9c8'
}

n_ctx_per_seq = 2048

class prompt_request(BaseModel):
    prompt: str
    model: str = "GGUF_model.gguf"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024


def cre_payload(prompt, model, temperature, top_p, max_tokens):

    return {
        'model': model,
        'messages': [{ 
            'role': 'system', 
            'content': 'You are B9 Assistant, a helpful and knowledgeable support bot for B9 users. You provide clear, accurate answers about B9 accounts, advances, loans, fees, direct deposits, and membership plans.' 
            },{ 
            'role': 'user', 
            'content': prompt
            }],
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
    }


async def send_request(payload):

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(url, headers=headers, data=json.dumps(payload))
    return response


async def process_request(request_data, future: asyncio.Future):
    async with semaphore:
    
        start_time = time.time()
        input_tokens = len(tokenizer.encode(request_data.prompt))

        if input_tokens + request_data.max_tokens > n_ctx_per_seq:
            allowed_input_tokens = n_ctx_per_seq - request_data.max_tokens
            if allowed_input_tokens <= 0:
                tokens = tokenizer.encode(request_data.prompt)[
                    :allowed_input_tokens]
                truncated_prompt = tokenizer.decode(tokens)
        else:
            truncated_prompt = request_data.prompt

        payload = cre_payload(
            truncated_prompt,
            request_data.model,
            request_data.temperature,
            request_data.top_p,
            request_data.max_tokens
        )

        response = await send_request(
            payload,
        )
        response_json = response.json()
        output_text = (
            response_json.get("choices", [{}])[0].get(
                "message", {}).get("content", "")
        )

        end_time = time.time()
        result = {
            "output_text": output_text,
            "time_taken": round(end_time - start_time, 2)
        }
        
        if not future.done():
            future.set_result(result)

def worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def worker_loop():
        while True:
            request_data, future = await loop.run_in_executor(None, request_queue.get)
            await process_request(request_data, future)
            request_queue.task_done()

    loop.create_task(worker_loop())
    loop.run_forever()

Thread(target=worker, daemon=True).start()

@app.post("/generate")
async def generate(request: prompt_request):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    request_queue.put((request, future))
    return await future