MathAI - Study Assistant MVP

This is a Streamlit-based MVP that demonstrates:

- PDF upload and text extraction (PyMuPDF)
- Chunking and embedding (sentence-transformers)
- Local vector store with Chroma
- RAG-based chat (OpenAI or fallback)
- Quiz generator (5 MCQs via LLM)
- Simple 7-day study plan generator
- Minimal teacher dashboard with fake data
- Urdu translation demo (MarianMT)

Setup

1. Create a Python virtual environment (recommended):

   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies:

   pip install -r requirements.txt

3. Set OPENAI_API_KEY environment variable if you want LLM-powered answers.

Run

   streamlit run app.py

Notes

- The app will create a local Chroma DB folder `chroma_db` when you index PDFs.
- If you don't have an OpenAI key, the app will still index and retrieve documents but LLM responses will be stubbed or simpler.
- For Urdu translation, the first run will download MarianMT model weights.

Next steps

- Add sample PDFs to test indexing.
- Wire production LLMs or a local Llama model for offline demos.
