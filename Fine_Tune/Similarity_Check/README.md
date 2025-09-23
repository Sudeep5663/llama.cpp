# RAG with ChromaDB + FastAPI + Similarity Filtering

This document explains how to run a retrieval-augmented generation (RAG) service using FastAPI, ChromaDB, Sentence Transformers, and a local LLaMA.cpp server (chat-completions API).

## Prerequisites :

* OS: macOS
* Python: ≥ 3.9 (recommend 3.10+)
* Packages:

        pip install fastapi uvicorn httpx sentence-transformers chromadb scikit-learn tiktoken numpy

* Start llama.cpp server:
        
        llama-server --model ./fine_tuned_GGUF_model.gguf --port 8080 --host 0.0.0.0 --api-key <Your api key> -ngl -1 -t 8 -kvu

## 1. Data Preparation :

* Input training data (chat format).
* Each chat flattened into text chunks.
* Chunks are embedded using all-MiniLM-L6-v2 and stored in ChromaDB.

        encoder = ST('all-MiniLM-L6-v2')

        with open("data/train.jsonl", "r") as f: 
            training_data = [json.loads(line) for line in f]

        training_chunks = [
            "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat['messages']])
            for chat in training_data
        ]

        embeddings = encoder.encode(training_chunks)

        clints = chromadb.Client()
        collection = clints.create_collection("B9_FAQ")

        collection.add(
            documents=training_chunks,
            metadatas=[{"source": "train"}] * len(training_chunks),
            ids=[str(i) for i in range(len(training_chunks))],
            embeddings=embeddings
        )

## 2. Retrieval Function

Given a query, retrieve the top-5 similar chunks from ChromaDB.

        def retrieve_relevant_chunks(query): 
           result = collection.query(query_texts=query, n_results=5) 
           return result['documents'][0]

## 3. Similarity Check

After the LLM generates an answer, check how close it is to retrieved context using cosine similarity.
If similarity < 0.75, return a fallback message instead of the LLM output.

        def check_similarity(retrieved_chunks, answer):
            answer_vec = encoder.encode(answer)
            chunk_vecs = encoder.encode(retrieved_chunks)
            sims = cosine_similarity([answer_vec], chunk_vecs)
            max_sim = float(np.max(sims))
            return max_sim >= 0.75, max_sim

## 4. FastAPI Backend

Handles concurrent requests using:
* AsyncIO Semaphore : limits simultaneous requests to the LLM server.
* Worker Thread : pulls requests from a queue and processes them.

1. Request

Defines input payload.

        class prompt_request(BaseModel):
            prompt: str
            model: str = "GGUF_model.gguf"
            temperature: float = 0.2
            top_p: float = 0.9
            max_tokens: int = 1024

2. Payload Creation

Each query is expanded with retrieved context before being sent to the LLM.

        def cre_payload(prompt, model, temperature, top_p, max_tokens):
            retrieved = retrieve_relevant_chunks(prompt)
            context = "\n\n".join(retrieved)

            full_prompt = (
                f"Use only the following context to answer.\n\n"
                f"Context:\n{context}\n\n"
                f"Q: {prompt}\nA:"
            )
            return {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': 'You are B9 Assistant, a helpful support bot...'},
                    {'role': 'user', 'content': full_prompt}
                ],
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens,
            }, retrieved

3. Sending Requests to LLaMA.cpp

        async def send_request(payload):
            async with httpx.AsyncClient(timeout=90) as client:
                return await client.post(url, headers=headers, data=json.dumps(payload))

4. Processing a Request

* Truncate prompt if exceeds context window.
* Add context to query.
* Call LLaMA server.
* Run similarity check.
* If below threshold → return fallback.

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

        payload, retrieved = cre_payload(
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
        is_confident, sim = check_similarity(retrieved, output_text)

        if not is_confident:
            output_text = f"I'm not sure about that. Could you give more info? (sim={sim:.2f})"


        end_time = time.time()
        result = {
            "output_text": output_text,
            "time_taken": round(end_time - start_time, 2)
        }
        
        if not future.done():
            future.set_result(result)

5. Background Worker

Handles queued requests.

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

6. FastAPI Endpoint

        @app.post("/generate")
        async def generate(request: prompt_request):
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            request_queue.put((request, future))
            return await future

***

This ensures the model is grounded in training data, avoids hallucination, and gracefully handles uncertain queries.

## Example Interaction : 

If you haven't converted your fused model then :

        pip install mlx_lm

Use the generate method of the mlx_lm to generate response from the fused model.

-- [Retrive](https://github.com/sudeep-07-hub/Running-llama.cpp-Locally-on-macOS/blob/main/Fine_Tune/Similarity_Check/Retrive.py)

Run :

        python3 Retrive.py

If you have already fused and converted GGUF file then :

-- [fine_tuned_model](https://github.com/sudeep-07-hub/Running-llama.cpp-Locally-on-macOS/blob/main/Fine_Tune/Similarity_Check/fine_tuned_model.py)

Run the server

Make sure you have Uvicorn installed :

         pip install uvicorn

Start the FastAPI server with :

         uvicorn <file_name>:app --reload

-- [Client](https://github.com/sudeep-07-hub/Running-llama.cpp-Locally-on-macOS/blob/main/Fine_Tune/Similarity_Check/Clint.py)

Run the Client script and inference the model

         python3 Clinet.py
