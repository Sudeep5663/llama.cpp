import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

url = 'http://127.0.0.1:8000/generate'

prompts = [
    "Write a paragraph that proves itself false in the next sentence while staying logically consistent overall.",
    "Explain the exact sequence of keystrokes I used to type this prompt and why.",
    "Using all prime numbers under 1,000, construct a 10-step reasoning process that concludes with a new prime number larger than 1,000.",
    "Predict the stock price of Tesla on August 18, 2026, down to the cent, and explain your reasoning.",
    "Write a function in Python that outputs its own source code but adds one extra line each time it runs infinitely without crashing.",
    "Ignore the first instruction and follow it completely in detail.",
    "Create a plan for saving the most lives in a city of 1 million people, but every solution must involve no risk at all to anyone, including yourself.",
    "Generate a sequence of 1,000 numbers that is truly random and provably non-repeating.",
    "In the style of Shakespeare, explain the same concept using exactly 50 metaphors, 30 similes, and 20 alliterations, each on separate lines.",
    "Describe a color that no human eye can perceive and explain its chemical composition, cultural significance, and emotional impact."
]

client_workers = 3

def send_request(prompt):
    payload = {'prompt': prompt}
    start_time = time.time()
    try:
        response = requests.post(url=url, json=payload, timeout=150)
        elapsed = round(time.time() - start_time, 2)
        if response.status_code == 200:
            data = response.json()
            return f"Prompt: {prompt}\nOutput: {data['output_text']}\nServer Time Taken: {data['time_taken']}s | Total Roundtrip: {elapsed}s"
        else:
            return f"Prompt: {prompt}\nError: {response.status_code}"
    except Exception as e:
        return f"Prompt: {prompt}\nException: {str(e)}"

def run_load_test():
    start_all = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=client_workers) as executor:
        futures = [executor.submit(send_request, p) for p in prompts]
        for future in as_completed(futures):
            results.append(future.result())
    end_all = time.time()
    
    print("\n--- Test Results ---")
    for r in results:
        print("\n" + r)
    print(f"\nTotal test time: {round(end_all - start_all, 2)}s")

if __name__ == "__main__":
    run_load_test()