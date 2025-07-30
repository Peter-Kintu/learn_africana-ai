from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import time
import httpx
import os
import google.generativeai as genai # Import the Gemini library

# ✅ Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TutorBot")

# ✅ FastAPI app
app = FastAPI(
    title="AI TutorBot",
    description="An AI tutor assistant API powered by Google Gemini",
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
# Use GEMINI_API_KEY instead of OPENROUTER_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the generative AI model
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-pro for text-only generation
    GEMINI_MODEL = genai.GenerativeModel('gemini-pro')
else:
    logger.error("GEMINI_API_KEY is not set. AI functionality will be disabled.")
    GEMINI_MODEL = None # Set to None if API key is missing

MIN_DELAY = 10.0 # Still keep rate limiting for your backend calls

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

# ✅ Ask Gemini
async def ask_gemini(prompt: str, student_id: str, retries: int = 3) -> str:
    if not GEMINI_MODEL:
        logger.error("Gemini model not configured due to missing API key.")
        raise HTTPException(status_code=500, detail="AI service not available: API key missing.")

    for attempt in range(1, retries + 1):
        try:
            # Use the configured Gemini model to generate content
            # The `generate_content` method sends the prompt to the model
            # Added generation_config to control max_output_tokens
            response = await asyncio.to_thread(
                GEMINI_MODEL.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000 # Increased from default/previous to allow longer responses
                )
            )
            
            # Access the text from the response
            # Gemini's response structure is different from OpenRouter's
            reply = response.text.strip()

            if not reply:
                logger.warning(f"Attempt {attempt}: Empty reply from Gemini for student {student_id}. Full response: {response}")
                if attempt == retries:
                    raise ValueError("Empty reply from Gemini after multiple attempts.")
                else:
                    await asyncio.sleep(2 ** attempt) # Exponential backoff
                    continue

            return reply

        except Exception as e:
            logger.error(f"Attempt {attempt} failed for student {student_id} with Gemini API: {e}")
            if attempt == retries:
                raise HTTPException(status_code=502, detail=f"TutorBot failed to respond from Gemini: {str(e)}")
            await asyncio.sleep(2 ** attempt) # Exponential backoff for all exceptions

# ✅ API endpoint
@app.post("/ask_tutor")
async def ask_tutor(request: TutorRequest):
    await wait_for_rate_limit()
    logger.info(f"Student: {request.student_id} | Subject: {request.subject} | Question: {request.question}")
    prompt = build_prompt(request.subject, request.level, request.question)
    answer = await ask_gemini(prompt, student_id=request.student_id) # Call the Gemini specific function

    # Provide a fallback message if the answer is still empty (should be caught by ask_gemini, but as a final safeguard)
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
