import os
import time
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
DATASET_NAME = "ai4bharat/MSMARCO-XI"
SUBSET_NAME = "default"  # Using default subset
COLLECTION_NAME = "msmarco_xi"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
MAX_SAMPLES = 500  # Limiting for quick setup during assignment

def init_qdrant():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        logger.info("Connecting to remote Qdrant...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    else:
        logger.info("Connecting to local in-memory Qdrant...")
        # Persist locally for reuse
        client = QdrantClient(path="./qdrant_data", timeout=60)
        
    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection {COLLECTION_NAME}")
    else:
        logger.info(f"Collection {COLLECTION_NAME} already exists.")
        
    return client

def chunk_and_embed():
    logger.info("Loading Embedding Model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Advanced Chunking Strategy: Recursive Character Splitting with Overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    client = init_qdrant()
    
    # We bypass the HuggingFace datasets library due to a known hanging bug with nested arrays in MSMARCO-XI streaming.
    # Instead, we directly fetch the validation Hindi partition (which has English pairs).
    parquet_filename = "hinval.parquet"
    parquet_url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
    
    if not os.path.exists(parquet_filename):
        logger.info(f"Downloading {parquet_filename} from HuggingFace (approx 461MB)...")
        import requests
        response = requests.get(parquet_url, stream=True)
        total_length = response.headers.get('content-length')
        
        with open(parquet_filename, "wb") as f:
            if total_length is None: # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                last_percent = 0
                for data in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    dl += len(data)
                    f.write(data)
                    percent = int(100 * dl / total_length)
                    if percent >= last_percent + 10:
                        logger.info(f"Download Progress: {percent}%")
                        last_percent = percent
        logger.info("Download complete!")
    
    logger.info(f"Loading dataset from {parquet_filename}...")
    import pandas as pd
    df = pd.read_parquet(parquet_filename).head(MAX_SAMPLES)
    
    points = []
    point_id = 1
    
    logger.info("Processing and Chunking data...")
    for index, row in df.iterrows():
        query_id = row.get("query_id", str(index))
        target_lang = row.get("target_lang", "hi")
        
        passages = row.get("passages", {})
        eng_passages = passages.get("English_passages", [])
        trans_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        for i, (eng, trans, sel) in enumerate(zip(eng_passages, trans_passages, is_selected)):
            # Chunk English passage
            eng_chunks = text_splitter.split_text(str(eng))
            for chunk in eng_chunks:
                points.append({
                    "id": point_id,
                    "text": chunk,
                    "metadata": {
                        "query_id": query_id,
                        "language": "en",
                        "is_selected": bool(sel),
                        "passage_index": i
                    }
                })
                point_id += 1
                
            # Chunk Translated passage
            trans_chunks = text_splitter.split_text(str(trans))
            for chunk in trans_chunks:
                points.append({
                    "id": point_id,
                    "text": chunk,
                    "metadata": {
                        "query_id": query_id,
                        "language": target_lang,
                        "is_selected": bool(sel),
                        "passage_index": i
                    }
                })
                point_id += 1
                
    logger.info(f"Generated {len(points)} chunks. Computing embeddings...")
    
    # Batch compute embeddings and upload
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        texts = [p["text"] for p in batch_points]
        embeddings = model.encode(texts).tolist()
        
        qdrant_points = []
        for j, p in enumerate(batch_points):
            # Combine text into metadata for retrieval context
            meta = p["metadata"]
            meta["text"] = p["text"]
            
            qdrant_points.append(
                PointStruct(
                    id=p["id"],
                    vector=embeddings[j],
                    payload=meta
                )
            )
            
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=qdrant_points
            )
        except Exception as e:
            logger.error(f"Failed to upsert batch {i} to {i + len(batch_points)}: {e}")
            continue

        if i % 500 == 0:
            logger.info(f"Uploaded {i}/{len(points)} chunks...")
            
    logger.info(f"Successfully ingested {len(points)} chunks into Qdrant.")

if __name__ == "__main__":
    start_time = time.time()
    chunk_and_embed()
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
