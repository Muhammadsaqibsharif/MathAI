# MathMind MVP - Adaptive Math Tutor

An intelligent math tutoring system that adapts to student performance and supports multiple languages.

## Features

- **Adaptive Learning**: Quiz difficulty and topic progression based on student performance
- **Multi-language Support**: English and Urdu translation capabilities
- **AI-Powered**: Uses AIMLAPI with Sonet 4.5 preview for explanations and quiz generation
- **Student Dashboard**: Interactive learning interface
- **Teacher Dashboard**: Progress tracking and analytics
- **RAG Integration**: Retrieval-Augmented Generation for contextual learning

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   copy .env.example .env
   # Edit .env and add your AIMLAPI_KEY
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Demo Setup**
   - Open http://localhost:8501
   - Select "Demo Runner" to create sample students
   - Switch to "Student" or "Teacher" mode to explore features

## Project Structure

```
mathmind-mvp/
├── data/               # Learning content
├── utils/              # Student data storage
├── app.py             # Main Streamlit application
├── ai_client.py       # AIMLAPI integration
├── rag_builder.py     # Embeddings and search
├── translate.py       # Translation services
├── adaptive.py        # Learning logic
├── storage.py         # Data persistence
└── requirements.txt   # Dependencies
```

## Environment Variables

- `AIMLAPI_KEY`: Your AIMLAPI key for AI services
- `AIMLAPI_BASE`: API base URL (default: https://api.aimlapi.com/v1)
- `DATA_PATH`: Path to learning content (default: ./data)
- `STORAGE_FILE`: Student data file (default: ./utils/sample_students.json)

## Demo Script

1. **Teacher Dashboard**: View student progress and analytics
2. **Student Mode**: 
   - Select topic (Fractions, Decimals, Algebra)
   - Get AI explanations in English/Urdu
   - Take adaptive quizzes
   - Receive personalized recommendations
3. **Demo Features**: Test AI responses and translations

## Dependencies

- Streamlit: Web interface
- AIMLAPI: AI responses via Sonet 4.5 preview
- Sentence Transformers: Text embeddings
- FAISS: Vector similarity search
- GoogleTrans: Translation services
- Plotly: Data visualization

## Fallback Features

- Demo responses when API keys are missing
- In-memory similarity search if FAISS fails
- Error handling for translation services
- Robust JSON parsing with fallbacks