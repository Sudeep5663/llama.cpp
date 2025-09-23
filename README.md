# Running llama.cpp Locally on macOS

This document walks through installing prerequisites, setting up llama.cpp via Homebrew, downloading a quantized model, and interacting with it from Python.

## Prerequisites :
* **Operating System**: macOS or Linux (works best on Apple Silicon for Metal acceleration).
* **Python**: ≥ 3.9 (recommended 3.10+).
* **Homebrew**: Required for macOS (to install dependencies like cmake).

## 1. Install Homebrew ( if not installed ) :

Check if home-brew is already installed :

    brew --version

If not :

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Verify installation : 
 
    brew --version

## 2. Install llama.cpp :

Use home-brew to install :

    brew install llama.cpp

This will install both the llama CLI and llama-sever binary.

Reference :

 [Homebrew Formulae]( https://formulae.brew.sh/formula/llama.cpp),
 [llama.cpp]( https://github.com/ggml-org/llama.cpp)

## 3. Run llama server with Hugging Face model :

llama-server supports loading models directly from Hugging Face without manual downloading. You just need to specify the repository and file name.

Example ([ mistralai/Mistral-7B-Instruct-v0.3 ](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) ) :

    llama-sever \
    --hf-repo MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF \
    --hf-file Mistral-7B-Instruct-v0.3.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --api-key <Your key> \
    -ngl -1 \
    -t 8 \
    -kvu

Flags explained :

`- -hf-repo —> Hugging face repo containing GGUF models.`

`- -hf-file —> Specific quantized model file to use ( e.g. Q4_K_M)`

`- -host —> Bind server ( use 0.0.0.0 for all interfaces).`

`- -port —> API server port.`

`- -api-key —> For authentication.`

`-ngl —> GPU layers to offload (-1 = auto for Metal on Mac).`

`-t —> Number of CPU threads (match your CPU cores).`

`-kvu —> Optimization flag for vocab/unified memory.`

LLaMA and similar models are very large (tens to hundreds of GB). To run them on a laptop, you typically use quantized versions.

**What is quantization?**

Quantization reduces the precision of model weights (e.g., from 16-bit → 4-bit) to save memory and make inference faster, with only a small loss in accuracy.

* FP16 (float16): full precision, largest size (50–100GB for 7B models).
* Q8/Q6/Q4 quantization: 8-bit, 6-bit, 4-bit compressed versions, much smaller (4–8GB).
* Q4_K_M → one of the most common balance points: small size, good quality.

Reference : 

[Toola]( https://github.com/ggml-org/llama.cpp/tree/master/tools/server),
[SteelPh0enix's Blog](https://blog.steelph0enix.dev/),
[llama_cpp](https://huggingface.co/docs/inference-endpoints/en/engines/llama_cpp),
[llama server settings](https://llama-cpp-python.readthedocs.io/en/latest/server/)

## 4. Query the API via Python :

Create a Python Client ( client.py ) :

-- [client](https://github.com/Sudeep5663/Running-llama.cpp-Locally-on-macOS/blob/main/test4.py)

Run :

    python3 client.py

**Notes and tips**

* For faster inference on Apple Silicon, ensure -ngl -1 to enable Metal GPU acceleration.
* If you want OpenAI API compatibility, the server already mimics /v1/completions and /v1/chat/completions.
* If storage is limited, stick to Q4_K_M OR Q5_K_M models ( smaller footprint).

**Example Interaction** :

Enter your prompt
    
    Explain gravity in simple terms

Response

    Gravity is a force that attracts two objects towards each other. It's the force that makes things fall to the ground, keeps planets orbiting the sun, and even holds the entire universe together!

## Concurrence and Load Testing with llama.cpp :

1. Handling Concurrent Requests with FastAPI

We creates a FastAPI wrapper around llama-server to :
* Queue incoming requests.
* Limit concurrent execution using semaphore.
* Truncate inputs if they exceed context length.
* Return structured output with timing info.

FastAPI Server ( server.py )

-- [server](https://github.com/Sudeep5663/llama.cpp/blob/main/server.py)

**Run the server**

Make sure you have Uvicorn installed :

    pip install uvicorn

Start the FastAPI server with :

    uvicorn <file_name>:app --reload

2. Sending Concurrent Requests

A client script that uses ThreadPoolExecutor to fire multiple prompts concurrently.

Client ( load_test.py )

-- [load_test](https://github.com/Sudeep5663/llama.cpp/blob/main/load_test.py)

Run :

    python3 load_test.py


**Observations :**

By default, llama-server in llama.cpp processes one request at a time. Even if you send multiple concurrent requests, the server will internally queue them and process sequentially. This has important implications for latency under load.

* Even with client_workers = 3, llama.cpp only processes one request at a time.
* Other requests are queued internally until the current request is finishes.
* If the front request takes too long, timeouts occur for others.
* Even running multiple llama-sever instance did not improve concurrency - only one instance works per request at a time.

**Key Takeway :**

* llama.cpp is optimized for single-stream inference, not true concurrent request processing.
* If you need true concurrent request handling with batching and GPU acceleration, use vLLM on Linux with NVIDIA GPUs.
* For CPU-only setups or Apple Silicon, continue with llama.cpp but expect requests to be processed sequentially.
