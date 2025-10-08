import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.aimlapi.com/v1")

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def _extract_response_text(resp) -> str:
    """Robustly extract text from various response shapes returned by the OpenAI client."""
    try:
        # object-like: resp.choices[0].message.content
        return resp.choices[0].message.content
    except Exception:
        try:
            return resp['choices'][0]['message']['content']
        except Exception:
            return str(resp)


def answer_with_context(question: str, context: str) -> str:
    """Use OpenAI chat completion to answer using provided context. If no key, return a simple extractive answer."""
    system_prompt = (
        "You are a helpful study assistant. Use the provided context strictly to answer the question. "
        "If the context doesn't contain the answer, say so and provide a short guidance on how to find it."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely using the context and cite source chunk indexes if helpful."

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0.2,
            )
            text = _extract_response_text(resp)
            return text.strip()
        except Exception as e:
            return f"OpenAI error: {e}\n\nFalling back to simple extraction.\n{context[:1000]}"
    else:
        snippet = context[:800]
        return f"(No OPENAI_API_KEY) Context snippet: {snippet}\n\nBased on the above, question: {question}"


def generate_quiz(topic_text: str) -> str:
    prompt = (
        "Generate 5 multiple choice questions (MCQs) from the following content. For each question, provide: "
        "question text, 4 options labelled A-D, the correct option letter, and a one-sentence explanation. Content:\n\n"
        + topic_text
        + "\n\nReturn plain text."
    )

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            text = _extract_response_text(resp)
            return text.strip()
        except Exception as e:
            return f"OpenAI error: {e}"
    else:
        return "(No OPENAI_API_KEY) Quiz generation requires an OpenAI key for full functionality."
