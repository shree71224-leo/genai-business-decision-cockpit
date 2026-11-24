import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# Load environment variables from the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# Gemini configuration
API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# Initialize Gemini client
client = (
    genai.Client(api_key=API_KEY)
    if genai and API_KEY
    else None
)