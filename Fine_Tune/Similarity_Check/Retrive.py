from sentence_transformers import SentenceTransformer as ST 
import json 
import chromadb 
import numpy as np 
from mlx_lm import load, generate 
from sklearn.metrics.pairwise import cosine_similarity 

encoder = ST("all-MiniLM-L6-v2") 

training_data = [] 

with open("data/train.jsonl", "r") as f: 
    for line in f.readlines(): training_data.append(json.loads(line)) 

training_chunks = [] 
for chat in training_data: 
    text = "\n".join( [f"{m['role'].capitalize()}: {m['content']}" for m in chat['messages']]) 
    training_chunks.append(text) 
    
embedding = encoder.encode(training_chunks) 

clints = chromadb.Client() 

collection = clints.create_collection("B9_FAQ") 
collection.add( 
    documents=training_chunks, 
    metadatas=[{"source": "train"}] * len(training_chunks), 
    ids=[str(i) for i in range(len(training_chunks))], 
    embeddings=embedding ) 

def retrieve_relevant_chunks(query): 
    result = collection.query( query_texts=query, n_results=5 ) 
    retrieved_chunks = [doc for doc in result['documents'][0]] 
    return retrieved_chunks 

model, tokenizer = load("/Users/sukesh/Desktop/Fine-Tune/fused_model") 

def generate_response(query):
    retrieved = retrieve_relevant_chunks(query)
    context = "\n\n".join(retrieved)
    final_prompt = (
        f"Use only the following context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {query}\nA:"
    )
    messages = [{"role": "user", "content": final_prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=300)
    return text, retrieved 

def check_similarity(retrieved_chunks, answer):
    answer_vec = encoder.encode(answer)
    doc_vecs = encoder.encode(retrieved_chunks)
    sims = cosine_similarity([answer_vec], doc_vecs)
    max_sim = np.max(sims)
    return max_sim >= 0.60, max_sim

def fallback(query):
    answer, retrieved = generate_response(query)
    is_confident, sim_value = check_similarity(retrieved, answer)
    if is_confident:
        return f"Answer: {answer.strip()} (sim={sim_value:.2f})"
    else:
        return f"I'm not sure about that. Could you give more info? (sim={sim_value:.2f})"
    
    
prompt = "Lost my B9 card, what should I do?" 
print(fallback(prompt))