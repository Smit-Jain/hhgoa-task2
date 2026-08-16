import os
import aiohttp
import logging
import time

logger = logging.getLogger(__name__)

async def sarvam_stt(audio_data: bytes) -> str:
    """
    Calls Sarvam AI's speech-to-text API asynchronously.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not set")
        
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": api_key
    }
    
    data = aiohttp.FormData()
    data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
    # Defaulting to Indian English or mixed language based on Sarvam's standard.
    # The user could change this to hi-IN depending on the actual input.
    data.add_field('language_code', 'en-IN') 
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("transcript", "")
            else:
                error_msg = await response.text()
                logger.error(f"Sarvam API Error: {error_msg}")
                raise Exception(f"Sarvam STT failed: {error_msg}")

async def elevenlabs_stt(audio_data: bytes) -> str:
    """
    Calls ElevenLabs speech-to-text API asynchronously.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set")
        
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key
    }
    
    data = aiohttp.FormData()
    data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
    data.add_field('model_id', 'eleven_english_v2') # Standard model ID
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("text", "")
            else:
                error_msg = await response.text()
                logger.error(f"ElevenLabs API Error: {error_msg}")
                raise Exception(f"ElevenLabs STT failed: {error_msg}")

async def process_audio(audio_data: bytes) -> str:
    provider = os.getenv("STT_PROVIDER", "sarvam").lower()
    start_time = time.time()
    
    try:
        if provider == "elevenlabs":
            text = await elevenlabs_stt(audio_data)
        else:
            text = await sarvam_stt(audio_data)
            
        latency = (time.time() - start_time) * 1000
        logger.info(f"STT Latency ({provider}): {latency:.2f}ms")
        return text
    except Exception as e:
        logger.error(f"STT Error: {e}")
        # Fallback mechanism can be implemented here if needed
        raise e
