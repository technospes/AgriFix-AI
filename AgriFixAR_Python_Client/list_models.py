# list_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))

try:
    models = genai.list_models()
    print("✅ Available models:")
    for model in models:
        if 'gemini' in model.name.lower():
            print(f"  📌 {model.name}")
            print(f"     Methods: {model.supported_generation_methods}")
except Exception as e:
    print(f"❌ Error: {e}")