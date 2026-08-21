import os
import time
import re
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
DATASET_NAME = "ai4bharat/MSMARCO-XI"
COLLECTION_NAME = "msmarco_xi"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
MAX_SAMPLES = 10  # Limiting for quick setup during assignment

class AdvancedSemanticChunker:
    def __init__(self, max_chars=400, overlap_sentences=1):
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        
    def split_into_sentences(self, text):
        # Semantic split by natural sentence boundaries (ignoring abbreviations when possible)
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text, query_id, language="en"):
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            
            # If a single sentence exceeds the limit, we have to forcefully split it (fallback)
            if sentence_len > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_length = 0
                
                # Naive split for the oversized sentence
                for j in range(0, sentence_len, self.max_chars):
                    chunks.append([sentence[j:j+self.max_chars]])
                continue

            if current_length + sentence_len <= self.max_chars:
                current_chunk.append(sentence)
                current_length += sentence_len + 1
            else:
                chunks.append(current_chunk)
                # Apply semantic overlap: keep the last N sentences for continuity
                overlap = current_chunk[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                current_chunk = overlap + [sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                
        if current_chunk:
            chunks.append(current_chunk)
            
        # Metadata Enrichment: Inject the Query ID context directly into the semantic space
        enriched_chunks = []
        for c in chunks:
            raw_text = " ".join(c)
            context_prefix = f"[Context for Query {query_id}] "
            enriched_chunks.append(context_prefix + raw_text)
            
        return enriched_chunks

def init_qdrant():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
        
    logger.info("Connecting to remote Qdrant Cloud...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
        
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection {COLLECTION_NAME}")
    return client

def chunk_and_embed():
    logger.info("Loading Embedding Model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    chunker = AdvancedSemanticChunker(max_chars=400, overlap_sentences=1)
    client = init_qdrant()
    
    parquet_filename = "hinval.parquet"
    parquet_url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
    
    if not os.path.exists(parquet_filename):
        logger.info(f"Downloading {parquet_filename} from HuggingFace (approx 461MB)...")
        import requests
        response = requests.get(parquet_url, stream=True)
        with open(parquet_filename, "wb") as f:
            for data in response.iter_content(chunk_size=1024*1024):
                f.write(data)
        logger.info("Download complete!")
    
    logger.info(f"Loading dataset from {parquet_filename}...")
    df = pd.read_parquet(parquet_filename).head(MAX_SAMPLES)
    
    points = []
    point_id = 1
    
    logger.info("Processing with Advanced Metadata-Aware Semantic Chunking...")
    for index, row in df.iterrows():
        query_id = row.get("query_id", str(index))
        target_lang = row.get("target_lang", "hi")
        
        passages = row.get("passages", {})
        eng_passages = passages.get("English_passages", [])
        trans_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        for i, (eng, trans, sel) in enumerate(zip(eng_passages, trans_passages, is_selected)):
            # Process English passage with advanced chunking
            eng_chunks = chunker.chunk_text(str(eng), query_id, language="en")
            for chunk in eng_chunks:
                points.append({
                    "id": point_id,
                    "text": chunk,
                    "metadata": {"query_id": query_id, "language": "en", "is_selected": bool(sel), "passage_index": i}
                })
                point_id += 1
                
            # Process Translated passage with advanced chunking
            trans_chunks = chunker.chunk_text(str(trans), query_id, language=target_lang)
            for chunk in trans_chunks:
                points.append({
                    "id": point_id,
                    "text": chunk,
                    "metadata": {"query_id": query_id, "language": target_lang, "is_selected": bool(sel), "passage_index": i}
                })
                point_id += 1
                
    logger.info(f"Generated {len(points)} highly-contextualized semantic chunks. Computing embeddings...")
    
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        texts = [p["text"] for p in batch_points]
        embeddings = model.encode(texts).tolist()
        
        qdrant_points = []
        for j, p in enumerate(batch_points):
            meta = p["metadata"]
            meta["text"] = p["text"]
            qdrant_points.append(PointStruct(id=p["id"], vector=embeddings[j], payload=meta))
            
        client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)
        if i % 500 == 0:
            logger.info(f"Uploaded {i}/{len(points)} chunks...")
            
    logger.info("Successfully ingested chunks into Qdrant.")

if __name__ == "__main__":
    start_time = time.time()
    chunk_and_embed()
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
