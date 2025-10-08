import os
import streamlit as st
from pathlib import Path
from src.utils.pdf_utils import extract_text, chunk_text
from src.utils.embedding_store import EmbeddingStore
from src.utils.qa import answer_with_context, generate_quiz
from src.utils.translator import translate_to_urdu
import json
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="MathAI Study Assistant")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

store = EmbeddingStore(persist_directory=str(BASE / "chroma_db"))

st.title("MathAI — Conversational Study Assistant (MVP)")

tabs = st.tabs(["Index PDF", "Chat (RAG)", "Quiz", "Study Plan", "Teacher Dashboard", "Urdu Demo"])

# Tab 1: Index PDF
with tabs[0]:
    st.header("Upload and index a textbook PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None:
        bytes_data = uploaded.read()
        st.write(f"Indexing: {uploaded.name}")
        text = extract_text(bytes_data)
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        ids = [f"{uploaded.name}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": uploaded.name, "chunk": i} for i in range(len(chunks))]
        store.index_documents(ids, chunks, metadatas)
        st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

    if st.button("Clear local vector DB"):
        store.delete_collection()
        st.warning("Local vector DB cleared")

# Tab 2: Chat RAG
with tabs[1]:
    st.header("Chat with the textbook (RAG)")
    user_q = st.text_area("Ask a question about indexed content:")
    top_k = st.slider("Retrieval k", 1, 10, 4)
    if st.button("Ask") and user_q.strip():
        docs = store.retrieve(user_q, k=top_k)
        context = "\n\n---\n\n".join([d['document'] for d in docs])
        answer = answer_with_context(user_q, context)
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Retrieved chunks"):
            for i, d in enumerate(docs):
                st.markdown(f"**{i+1}.** Source: {d['metadata'].get('source')} (chunk {d['metadata'].get('chunk')})")
                st.write(d['document'])

# Tab 3: Quiz
with tabs[2]:
    st.header("Generate 5 MCQs from a topic")
    topic = st.text_area("Paste topic text (or a retrieved chunk):")
    if st.button("Generate Quiz") and topic.strip():
        quiz_text = generate_quiz(topic)
        st.text_area("Generated MCQs", value=quiz_text, height=400)

# Tab 4: Study Plan
with tabs[3]:
    st.header("Auto 7-day study plan")
    exam_date = st.date_input("Exam date (optional)")
    hours_per_day = st.number_input("Hours available per day", min_value=0.0, max_value=24.0, value=2.0)
    if st.button("Generate Plan"):
        # Simple heuristic planner: get top topics from DB
        topics = store.list_top_sources(limit=10)
        # create a 7-day plan
        days = 7
        plan = []
        for d in range(days):
            topic = topics[d % max(1, len(topics))]
            plan.append({"day": d+1, "topic": topic, "task": f"Read and solve exercises from {topic} for {hours_per_day} hours"})
        st.subheader("7-day Plan")
        for p in plan:
            st.write(f"Day {p['day']}: {p['topic']} — {p['task']}")

# Tab 5: Teacher Dashboard
with tabs[4]:
    st.header("Teacher Dashboard (minimal)")
    # Load sample fake data
    sample = DATA_DIR / "sample_past_papers.json"
    if sample.exists():
        with open(sample, 'r', encoding='utf8') as f:
            past = json.load(f)
    else:
        past = []
    st.metric("Class", "Class 9 - Physics")
    students = [
        {"name": "Ali", "progress": 0.72},
        {"name": "Sara", "progress": 0.85},
        {"name": "Ayesha", "progress": 0.58},
    ]
    cols = st.columns(len(students))
    for c, s in zip(cols, students):
        c.subheader(s['name'])
        c.progress(s['progress'])
    st.subheader("Generated Quizzes")
    st.write("(Sample) Quiz sessions and scores")
    st.table([
        {"quiz": "Chap 1 MCQ", "average": "78%"},
        {"quiz": "Chap 2 MCQ", "average": "65%"},
    ])

# Tab 6: Urdu Demo
with tabs[5]:
    st.header("Urdu Translation Demo")
    summary = st.text_area("Paste English summary to translate to Urdu:")
    if st.button("Translate to Urdu") and summary.strip():
        ur = translate_to_urdu(summary)
        st.write(ur)

st.sidebar.markdown("---")
st.sidebar.write("MathAI MVP — streamlit app. Provide OPENAI_API_KEY if you want LLM-powered answers.")
