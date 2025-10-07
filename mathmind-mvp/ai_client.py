# ai_client.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()

AIMLAPI_KEY = os.getenv("AIMLAPI_KEY")
AIMLAPI_BASE = os.getenv("AIMLAPI_BASE", "https://api.aimlapi.com/v1")
MODEL_NAME = "sonet-4.5-preview"   # chosen model

HEADERS = {
    "Authorization": f"Bearer {AIMLAPI_KEY}",
    "Content-Type": "application/json"
}

def get_llm_response(prompt, system_prompt="You are a friendly grade 6-8 math tutor.", temperature=0.2, max_tokens=800):
    if not AIMLAPI_KEY:
        # Fallback response when API key is missing
        return f"[Demo Mode] This is a placeholder response for: {prompt[:100]}... Please set AIMLAPI_KEY in your .env file to get real AI responses."
    
    url = f"{AIMLAPI_BASE}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # adapt depending on AIMLAPI response structure
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback for any API errors
        return f"[Error] Unable to connect to AI service. Demo response for: {prompt[:50]}..."