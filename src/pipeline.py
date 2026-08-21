import os
import json
import time
import logging
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq
from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding
import hashlib

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Groq client
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model using fastembed (ONNX-based, ultra-lightweight for Serverless/Vercel)
logger.info("Loading fastembed model for pipeline...")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Qdrant async client (Strictly Cloud)
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")

qclient = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
COLLECTION_NAME = "msmarco_xi"

# Fast In-Memory Semantic Cache
SEMANTIC_CACHE = {}

SYSTEM_PROMPT = """You are a helpful retrieval-augmented AI assistant. 
You are provided with a set of retrieved context passages from a large knowledge base.
Your goal is to answer the user's question using the provided context as your primary source.

RULES:
1. Use the provided context to formulate your answer. You may synthesize information across multiple passages.
2. If the context contains information even partially related to the query, answer to the best of your ability using that context. Set hallucination_detected to false.
3. Only set hallucination_detected to true if the context is completely irrelevant and you cannot provide any meaningful answer from it.
4. If the user's query is explicitly harmful, toxic, or dangerous (e.g., asking how to harm someone), set is_safe to false.
5. If the query has absolutely zero connection to any of the provided passages, set is_off_topic to true.

You MUST respond with valid JSON matching this schema:
{
  "answer": "Your answer string",
  "is_safe": true,
  "is_off_topic": false,
  "hallucination_detected": false
}
"""

async def retrieve_context(query: str, top_k: int = 3) -> list:
    """
    Retrieves the most relevant chunks from Qdrant using dense vector search.
    """
    start_time = time.time()
    embeddings = list(embedder.embed([query]))
    vector = embeddings[0].tolist()
    
    try:
        results = await qclient.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k
        )
        contexts = [hit.payload.get("text", "") for hit in results.points if hit.payload]
        logger.info(f"Retrieval Latency: {(time.time() - start_time) * 1000:.2f}ms")
        return contexts
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

async def generate_response(query: str, contexts: list, retries: int = 3) -> dict:
    """
    Calls the LLM (Groq) with structured JSON output and retry logic.
    """
    context_str = "\n---\n".join(contexts)
    prompt = f"Context:\n{context_str}\n\nUser Query: {query}"
    
    for attempt in range(retries):
        try:
            start_time = time.time()
            chat_completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-20b",
                response_format={"type": "json_object"},
                temperature=0.1
            )
            latency = (time.time() - start_time) * 1000
            logger.info(f"Groq LLM Latency: {latency:.2f}ms")
            
            result_str = chat_completion.choices[0].message.content
            return json.loads(result_str)
            
        except Exception as e:
            logger.error(f"Error calling Groq (Attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                return {
                    "answer": "An error occurred while communicating with the LLM.",
                    "is_safe": True,
                    "is_off_topic": False,
                    "hallucination_detected": False
                }
            await asyncio.sleep(0.1)

def get_cache_key(query: str) -> str:
    # Lowercase and strip for better semantic normalization
    normalized = query.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

async def run_pipeline(query: str) -> dict:
    """
    Orchestrates the retrieval and generation, applying guardrails to the final output.
    """
    start_time = time.time()
    if not query.strip():
        return {"error": "Empty query."}
        
    cache_key = get_cache_key(query)
    if cache_key in SEMANTIC_CACHE:
        logger.info(f"Cache HIT for query: {query}")
        result = SEMANTIC_CACHE[cache_key]
        total_latency = (time.time() - start_time) * 1000
        result["total_latency_ms"] = round(total_latency, 2)
        result["cached"] = True
        return result
        
    contexts = await retrieve_context(query)
    
    if not contexts:
        return {
            "query": query,
            "answer": "I could not find any relevant information to answer your question.",
            "total_latency_ms": (time.time() - start_time) * 1000
        }
        
    result = await generate_response(query, contexts)
    
    # Enforce Guardrails: only hard-block genuinely unsafe queries
    if not result.get("is_safe", True):
        final_answer = "This query violates safety policies. I cannot answer it."
    else:
        final_answer = result.get("answer", "No answer generated.")
        
    total_latency = (time.time() - start_time) * 1000
    
    final_output = {
        "query": query,
        "answer": final_answer,
        "retrieved_contexts": contexts,
        "metadata": {
            "is_safe": result.get("is_safe"),
            "is_off_topic": result.get("is_off_topic"),
            "hallucination_detected": result.get("hallucination_detected")
        },
        "total_latency_ms": round(total_latency, 2)
    }
    
    # Populate the Semantic Cache
    SEMANTIC_CACHE[cache_key] = final_output
    return final_output
