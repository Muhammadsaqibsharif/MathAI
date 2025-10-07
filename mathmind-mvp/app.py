# app.py
import streamlit as st
from rag_builder import load_text_chunks
from translate import detect_lang, to_english, to_urdu
from adaptive import generate_quiz, next_topic_logic
from storage import load_data, update_student_result
import json, os
from ai_client import get_llm_response

st.set_page_config(page_title="MathMind MVP", layout="wide")
st.title("MathMind — Adaptive Math Tutor (Demo)")

# Sidebar: user type
user_type = st.sidebar.selectbox("I am a", ["Student", "Teacher", "Demo Runner"])
lang = st.sidebar.radio("Language", ["English", "Urdu"])
data = load_data()

if user_type == "Student":
    st.header("Student Mode")
    student_id = st.text_input("Student ID", value="student_1")
    topic = st.selectbox("Choose topic", ["Fractions", "Decimals", "Algebra - Basics"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Explain topic"):
            with st.spinner("Getting explanation..."):
                prompt = f"Explain {topic} for grade 6 students with simple examples."
                # if user requested Urdu mode, translate workflow
                if lang == "Urdu":
                    # generate in English then translate
                    eng_resp = get_llm_response(prompt)
                    out = to_urdu(eng_resp)
                else:
                    out = get_llm_response(prompt)
                st.markdown(out)
    
    with col2:
        if st.button("Get 5-question quiz"):
            with st.spinner("Generating quiz..."):
                qns = generate_quiz(topic, n_questions=5)
                st.session_state["quiz"] = qns
                st.session_state["topic"] = topic
                st.success("Quiz generated! Scroll down to take it.")

    if "quiz" in st.session_state:
        st.subheader(f"Quiz: {st.session_state.get('topic')}")
        answers = []
        for i, q in enumerate(st.session_state["quiz"]):
            st.write(f"**Q{i+1}.** {q['q']}")
            # show options
            choice = st.radio(
                f"Select answer for Q{i+1}:", 
                options=q.get("options", ["A", "B", "C", "D"]), 
                key=f"q{i}"
            )
            answers.append(q.get("options", []).index(choice) if choice in q.get("options", []) else 0)
        
        if st.button("Submit Quiz"):
            correct = 0
            for i, q in enumerate(st.session_state["quiz"]):
                if answers[i] == q.get("answer", 0):
                    correct += 1
            score_pct = int((correct/len(st.session_state["quiz"]))*100)
            st.success(f"Score: {score_pct}% ({correct}/{len(st.session_state['quiz'])})")
            update_student_result(student_id, st.session_state["topic"], score_pct)
            
            # compute next topic (simple demo curriculum)
            curriculum = ["Fractions", "Decimals", "Algebra - Basics"]
            next_t = next_topic_logic(score_pct, st.session_state["topic"], curriculum)
            
            if score_pct >= 80:
                st.balloons()
                st.success(f"Excellent work! Recommended next topic: **{next_t}**")
            elif score_pct >= 50:
                st.info(f"Good effort! Practice more with: **{next_t}**")
            else:
                st.warning(f"Let's review the basics. Recommended topic: **{next_t}**")

elif user_type == "Teacher":
    st.header("Teacher Dashboard")
    students = data.get("students", [])
    if not students:
        st.info("No students yet. Use Demo Runner to create sample students.")
    else:
        import pandas as pd
        import plotly.express as px
        
        # Create summary table
        rows = []
        for s in students:
            history = s.get("history", [])
            if history:
                last = history[-1]
                total_attempts = len(history)
                avg_score = sum(h.get("score", 0) for h in history) / len(history)
                rows.append({
                    "Student ID": s["id"], 
                    "Last Topic": last.get("topic", "N/A"), 
                    "Last Score": f"{last.get('score', 0)}%",
                    "Total Attempts": total_attempts,
                    "Average Score": f"{avg_score:.1f}%"
                })
            else:
                rows.append({
                    "Student ID": s["id"], 
                    "Last Topic": "No attempts", 
                    "Last Score": "0%",
                    "Total Attempts": 0,
                    "Average Score": "0%"
                })
        
        df = pd.DataFrame(rows)
        st.subheader("Student Overview")
        st.dataframe(df, use_container_width=True)
        
        if not df.empty and len([r for r in rows if r["Total Attempts"] > 0]) > 0:
            # Create visualization
            chart_data = []
            for s in students:
                history = s.get("history", [])
                if history:
                    last_score = history[-1].get("score", 0)
                    chart_data.append({"Student": s["id"], "Last Score": last_score})
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig = px.bar(chart_df, x="Student", y="Last Score", 
                           title="Latest Quiz Scores by Student",
                           color="Last Score",
                           color_continuous_scale="viridis")
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed history
        st.subheader("Detailed Student History")
        selected_student = st.selectbox("Select student for detailed view:", 
                                      ["All"] + [s["id"] for s in students])
        
        if selected_student != "All":
            student_data = next((s for s in students if s["id"] == selected_student), None)
            if student_data and student_data.get("history"):
                history_df = pd.DataFrame(student_data["history"])
                st.write(f"**{selected_student}'s Performance History:**")
                st.dataframe(history_df, use_container_width=True)
                
                # Progress chart
                if len(history_df) > 1:
                    fig = px.line(history_df.reset_index(), x="index", y="score", 
                                title=f"{selected_student}'s Progress Over Time",
                                markers=True)
                    fig.update_xaxis(title="Attempt Number")
                    fig.update_yaxis(title="Score (%)")
                    st.plotly_chart(fig, use_container_width=True)

elif user_type == "Demo Runner":
    st.header("Demo Setup & Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Data")
        if st.button("Create 3 sample students"):
            data = load_data()
            data["students"] = [
                {"id":"student_1","history":[{"topic":"Fractions","score":45}]},
                {"id":"student_2","history":[{"topic":"Fractions","score":85}]},
                {"id":"student_3","history":[{"topic":"Decimals","score":70}]}
            ]
            from storage import save_data
            save_data(data)
            st.success("✅ Created sample students with initial scores!")
        
        if st.button("Clear all student data"):
            data = {"students": []}
            from storage import save_data
            save_data(data)
            st.success("🗑️ Cleared all student data!")
    
    with col2:
        st.subheader("System Test")
        if st.button("Test AI Response"):
            with st.spinner("Testing AI client..."):
                test_response = get_llm_response("What is 2+2? Answer briefly.")
                st.write("**AI Response:**")
                st.write(test_response)
        
        if st.button("Test Translation"):
            with st.spinner("Testing translation..."):
                test_english = "Mathematics is fun and important for learning."
                test_urdu = to_urdu(test_english)
                st.write("**English:** " + test_english)
                st.write("**Urdu:** " + test_urdu)
    
    # System status
    st.subheader("System Status")
    
    # Check environment
    import os
    api_key_status = "✅ Set" if os.getenv("AIMLAPI_KEY") else "❌ Missing"
    st.write(f"**AIMLAPI Key:** {api_key_status}")
    
    # Check dependencies
    try:
        import faiss
        faiss_status = "✅ Available"
    except ImportError:
        faiss_status = "⚠️ Not available (using fallback)"
    st.write(f"**FAISS:** {faiss_status}")
    
    data_files = os.path.exists("./data/chapter1.txt")
    st.write(f"**Sample Data:** {'✅ Available' if data_files else '❌ Missing'}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MathMind MVP** - Adaptive Math Learning")
st.sidebar.markdown("Built with Streamlit & AI")