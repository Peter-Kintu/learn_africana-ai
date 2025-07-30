from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import time
import httpx
import os

# ✅ Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TutorBot")

# ✅ FastAPI app
app = FastAPI(
    title="AI TutorBot",
    description="An AI tutor assistant API powered by OpenRouter",
    version="1.0"
)

# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Input schema
class TutorRequest(BaseModel):
    student_id: str
    subject: str
    level: str
    question: str

# ✅ Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistral-7b-openorca"  # ✅ Free tier model
MIN_DELAY = 10.0

# ✅ Global rate limiter
app.state.api_lock = asyncio.Lock()
app.state.last_call_time = 0.0

async def wait_for_rate_limit():
    async with app.state.api_lock:
        now = time.time()
        elapsed = now - app.state.last_call_time
        wait_time = max(0, MIN_DELAY - elapsed)
        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.2f}s...")
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
        f"Please explain in a clear, friendly tone with examples."
    )

# ✅ Ask OpenRouter
async def ask_openrouter(prompt: str, student_id: str, retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        logger.error("Missing OpenRouter API Key.")
        raise HTTPException(status_code=500, detail="Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-tutorbot.app",  # Optional for OpenRouter
        "X-Title": "AI TutorBot"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 200,
        "user": student_id
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if not reply:
                    logger.warning(f"Attempt {attempt}: Empty reply from OpenRouter for student {student_id}. Full response: {data}")
                    # If it's the last attempt and still no reply, raise an error
                    if attempt == retries:
                        raise ValueError("Empty reply from OpenRouter after multiple attempts.")
                    else:
                        # Continue to next retry if reply is empty but not last attempt
                        await asyncio.sleep(2 ** attempt) # Exponential backoff
                        continue

                return reply

            except httpx.HTTPStatusError as e:
                logger.error(f"Status error {e.response.status_code}: {e.response.text} for student {student_id}")
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"Rate limited. Retrying in {retry_after}s...")
                    await asyncio.sleep(retry_after)
                elif attempt == retries:
                    raise HTTPException(status_code=502, detail=f"Failed to get response from the tutor: {e.response.text}")
            except Exception as e:
                logger.error(f"Attempt {attempt} failed for student {student_id}: {e}")
                if attempt == retries:
                    raise HTTPException(status_code=502, detail=f"TutorBot failed to respond: {str(e)}")
                # Exponential backoff for other exceptions before retrying
                await asyncio.sleep(2 ** attempt)

# ✅ API endpoint
@app.post("/ask_tutor")
async def ask_tutor(request: TutorRequest):
    await wait_for_rate_limit()
    logger.info(f"Student: {request.student_id} | Subject: {request.subject} | Question: {request.question}")
    prompt = build_prompt(request.subject, request.level, request.question)
    answer = await ask_openrouter(prompt, student_id=request.student_id)

    # Provide a fallback message if the answer is still empty (should be caught by ask_openrouter, but as a final safeguard)
    if not answer:
        answer = "Apologies, the TutorBot couldn’t generate a reply. Please try again shortly."
        logger.warning(f"Final answer was empty for student {request.student_id}. Using fallback message.")

    return {
        "student_id": request.student_id,
        "subject": request.subject,
        "level": request.level,
        "question": request.question,
        "answer": answer
    }
