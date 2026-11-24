
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None

# Load local environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Get API key from local environment or Streamlit Cloud Secrets
try:
    API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini model configuration
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
