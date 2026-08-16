# HH Goa Task 2 - Voice-Enabled RAG System

This repository contains an end-to-end Voice-Enabled RAG pipeline built for Hacker House Goa 2026.

## Features
- **Voice-to-Text**: Integrates Sarvam or ElevenLabs STT.
- **Advanced Chunking**: Recursive character splitting with overlapping and metadata injection.
- **Fast Retrieval**: In-memory Qdrant.
- **Blazing Generation**: Groq LPU inference for sub-100ms LLM calls.
- **Guardrails**: Prompt-based hallucination detection, safety checks, and off-topic rejection.
- **Clean UI**: Dark-themed vanilla JS frontend mimicking HH Goa style.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Rename `.env.example` to `.env` and fill in your API keys:
```env
GROQ_API_KEY=your_groq_key
SARVAM_API_KEY=your_sarvam_key
# or ELEVENLABS_API_KEY=your_elevenlabs_key
```

### 3. Data Ingestion
Run the ingestion script to download `ai4bharat/MSMARCO-XI`, chunk it, embed it, and load it into Qdrant.
```bash
python data_ingestion.py
```
*(Note: It takes a subset of the first 500 samples for fast initialization during testing).*

### 4. Start the Application
Run the FastAPI backend.
```bash
uvicorn app:app --reload
```
Navigate to `http://localhost:8000` in your browser.

### 5. Latency Analytics
To benchmark the retrieval and generation pipeline:
```bash
python latency_analytics.py
```
This script will output the **P50**, **P70**, and **P100** latency percentiles, verifying the `< 200ms` strict requirement.
