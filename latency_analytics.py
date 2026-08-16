import asyncio
import time
import numpy as np
from src.pipeline import run_pipeline

# Sample queries from MSMARCO or generic topics
test_queries = [
    "What is the definition of a cell?",
    "How many bones are in the human body?",
    "What causes earthquakes?",
    "Who is the author of Harry Potter?",
    "What is the speed of light?",
    "Explain the theory of relativity.",
    "What are the symptoms of COVID-19?",
    "How do airplanes fly?",
    "What is the currency of Japan?",
    "Who painted the Mona Lisa?"
]

async def run_benchmark():
    latencies = []
    
    print(f"Starting latency benchmarking for {len(test_queries)} queries...")
    print("-" * 50)
    
    # Warmup query (since first load of embeddings/models can be slower)
    await run_pipeline("warmup query")
    
    for i, q in enumerate(test_queries):
        start_time = time.time()
        res = await run_pipeline(q)
        
        # Calculate latency just for retrieval + generation (ignoring network hop)
        latency = res.get("total_latency_ms", 0)
        if latency == 0:
            # Fallback if the pipeline didn't return latency
            latency = (time.time() - start_time) * 1000
            
        latencies.append(latency)
        print(f"[{i+1}/{len(test_queries)}] Query: '{q}' -> {latency:.2f}ms")
        
    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p100 = np.percentile(latencies, 100)
    
    print("-" * 50)
    print("🚀 Latency Benchmark Results (Retrieval + Generation):")
    print(f"Target: < 200ms")
    print("-" * 50)
    print(f"P50 Latency:  {p50:.2f} ms")
    print(f"P70 Latency:  {p70:.2f} ms")
    print(f"P100 Latency: {p100:.2f} ms")
    
    if p100 < 200:
        print("✅ SUCCESS: System meets the aggressive < 200ms latency requirement.")
    else:
        print("⚠️ WARNING: P100 exceeded 200ms. Consider moving Qdrant to local memory, using Groq API, or a smaller embedding model.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
