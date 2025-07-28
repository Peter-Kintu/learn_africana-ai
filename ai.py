from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import asyncio
import time
import httpx
import os

#
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TutorBot")

# ✅ FastAPI app instance
app = FastAPI(
    title="AI TutorBot",
    description="An AI tutor assistant API powered by OpenRouter",
    version="1.0"
)

# ✅ Request schema
class TutorRequest(BaseModel):
    student_id: str
    subject: str    # e.g., math, science, coding
    level: str      # e.g., beginner, intermediate, advanced
    question: str

# ✅ Configuration
# Use a safer way to load the API key (don't hardcode in production)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set this in your .env or environment
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/auto"
MIN_DELAY = 10.0  # seconds between global requests

# ✅ Global rate limiter
app.state.api_lock = asyncio.Lock()
app.state.last_call_time = 0.0

async def wait_for_rate_limit():
    async with app.state.api_lock:
        now = time.time()
        elapsed = now - app.state.last_call_time
        wait_time = max(0, MIN_DELAY - elapsed)
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.2f}s due to rate limiting...")
            await asyncio.sleep(wait_time)
        app.state.last_call_time = time.time()

# ✅ Prompt builder
def build_prompt(subject: str, level: str, question: str) -> str:
    prompts = {
        "math": "You are a friendly Math tutor.",
        "science": "You explain science with clarity and excitement.",
        "sst": "You teach Social Studies using relatable examples.",
        "english": "You help students learn English using simple grammar tips and vocabulary.",
        "biology": "You explain biology concepts visually and clearly.",
        "chemistry": "You explain chemistry using everyday examples.",
        "coding": "You teach programming with clear code samples.",
        "history": "You explain history as engaging stories."
    }

    intro = prompts.get(subject.lower(), "You are a helpful AI tutor.")
    return (
        f"{intro} The student is at a {level} level.\n"
        f'They asked: "{question}"\n'
        f"Please explain in a clear, friendly tone with examples to help them understand."
    )

# ✅ OpenRouter API logic
async def ask_openrouter(prompt: str, student_id: str, retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OpenRouter API key")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://your-tutorbot.app",  # Optional for tracking
        "X-Title": "AI TutorBot"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 200,
        "user": student_id  # ✅ Helps OpenRouter track usage per user
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"Rate limited by OpenRouter. Retrying in {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if not reply:
                    raise ValueError("Empty response from OpenRouter")

                return reply

            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise HTTPException(status_code=502, detail="Failed to get response from AI Tutor")

# ✅ Main endpoint
@app.post("/ask_tutor")
async def ask_tutor(request: TutorRequest):
    await wait_for_rate_limit()

    logger.info(f"📩 Received question from student {request.student_id} on subject: {request.subject}")
    prompt = build_prompt(request.subject, request.level, request.question)
    response = await ask_openrouter(prompt, student_id=request.student_id)

    return {
        "student_id": request.student_id,
        "subject": request.subject,
        "level": request.level,
        "question": request.question,
        "answer": response
    }
