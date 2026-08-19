import os
import json
import time
import logging
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Groq client
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model globally to keep it in memory for < 200ms latency
logger.info("Loading embedding model for pipeline...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant async client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
if qdrant_url and qdrant_api_key:
    qclient = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
else:
    # Using local path for fast retrieval during testing
    qclient = AsyncQdrantClient(path="./qdrant_data")
    
COLLECTION_NAME = "msmarco_xi"

SYSTEM_PROMPT = """You are a highly precise retrieval-augmented AI assistant. 
You are provided with a set of retrieved context passages from the MSMARCO-XI dataset.
Your goal is to answer the user's question STRICTLY using the provided context.

GUARDRAILS:
1. Grounding: Do not use any outside knowledge. If the context does not contain the answer, you must admit you don't know and set hallucination_detected to true.
2. Topic: If the user's query is completely unrelated to any provided context, set is_off_topic to true and refuse to answer.
3. Safety: If the query is harmful, toxic, or unsafe, set is_safe to false and refuse to answer.

You MUST respond with valid JSON matching this schema:
{
  "answer": "Your answer string",
  "is_safe": true/false,
  "is_off_topic": true/false,
  "hallucination_detected": true/false
}
"""

async def retrieve_context(query: str, top_k: int = 3) -> list:
    """
    Retrieves the most relevant chunks from Qdrant using dense vector search.
    """
    start_time = time.time()
    vector = embedder.encode(query).tolist()
    
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

async def run_pipeline(query: str) -> dict:
    """
    Orchestrates the retrieval and generation, applying guardrails to the final output.
    """
    start_time = time.time()
    if not query.strip():
        return {"error": "Empty query."}
        
    contexts = await retrieve_context(query)
    
    if not contexts:
        return {
            "query": query,
            "answer": "I could not find any relevant information to answer your question.",
            "total_latency_ms": (time.time() - start_time) * 1000
        }
        
    result = await generate_response(query, contexts)
    
    # Enforce Guardrails strictly
    if not result.get("is_safe", True):
        final_answer = "This query violates safety policies. I cannot answer it."
    elif result.get("is_off_topic", False):
        final_answer = "This query is off-topic based on the available context."
    elif result.get("hallucination_detected", False):
        final_answer = "The context does not contain enough information to answer reliably without hallucinating."
    else:
        final_answer = result.get("answer", "No answer generated.")
        
    total_latency = (time.time() - start_time) * 1000
    
    return {
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
