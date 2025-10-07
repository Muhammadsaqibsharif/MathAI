# adaptive.py
from ai_client import get_llm_response

def generate_quiz(topic_name, n_questions=5, difficulty="medium"):
    """
    Ask the LLM (Sonet) to generate n MCQs for the topic.
    We return a list of {question, options, answer_index}
    """
    prompt = f"""Generate {n_questions} multiple choice questions for grade 6-8 on the topic '{topic_name}'.
Each question should have 4 options. Provide the answer index (0-3) at the end in JSON array format only.
Difficulty: {difficulty}.
Return output as valid JSON array like:
[{{"q":"What is 1/2 + 1/4?","options":["1/6","3/4","2/6","1/8"],"answer":1}}, ...]"""
    
    resp = get_llm_response(prompt)
    
    # Try to parse JSON response
    import json
    try:
        return json.loads(resp)
    except Exception as e:
        # Fallback: return sample questions if parsing fails
        fallback_questions = [
            {
                "q": f"Sample question about {topic_name}: What is the basic concept?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": 0
            },
            {
                "q": f"Another {topic_name} question: Which is correct?",
                "options": ["Choice 1", "Choice 2", "Choice 3", "Choice 4"],
                "answer": 1
            },
            {
                "q": f"Final {topic_name} question: What applies here?",
                "options": ["Answer A", "Answer B", "Answer C", "Answer D"],
                "answer": 2
            }
        ]
        return fallback_questions[:n_questions]

def next_topic_logic(last_score_pct, current_topic, curriculum_order):
    """
    Simple logic:
    score >=80 -> advance to next
    50-79 -> remain same (practice)
    <50 -> go back or provide remedial
    """
    try:
        idx = curriculum_order.index(current_topic)
        if last_score_pct >= 80:
            return curriculum_order[min(idx+1, len(curriculum_order)-1)]
        elif last_score_pct < 50:
            return curriculum_order[max(idx-1, 0)]
        else:
            return current_topic
    except ValueError:
        # If topic not in curriculum, return first topic
        return curriculum_order[0] if curriculum_order else current_topic