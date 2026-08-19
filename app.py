import os
import time
import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# MUST load dotenv before importing src modules so they have access to API keys!
load_dotenv()

from src.stt import process_audio
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/ask")
async def ask_question(audio: UploadFile = File(...), override_query: str = Form(None)):
    """
    Endpoint to receive voice input, transcribe it, and return a grounded RAG response.
    """
    start_time = time.time()
    
    # 1. Speech-to-Text (or prompt chip override)
    if override_query and override_query.strip():
        query = override_query.strip()
    else:
        audio_data = await audio.read()
        try:
            query = await process_audio(audio_data)
        except Exception as e:
            return JSONResponse({"error": f"STT Provider Error: {str(e)}"}, status_code=500)
    
    if not query:
        return JSONResponse({"error": "Failed to transcribe audio. Ensure audio is clear."}, status_code=400)
        
    logger.info(f"Transcribed query: {query}")
    
    # 2. Retrieval & Generation
    rag_result = await run_pipeline(query)
    
    total_time = (time.time() - start_time) * 1000
    rag_result["total_end_to_end_latency_ms"] = round(total_time, 2)
    
    return JSONResponse(rag_result)
