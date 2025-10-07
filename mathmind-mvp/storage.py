# storage.py
import json
import os

DEFAULT_PATH = os.getenv("STORAGE_FILE", "./utils/sample_students.json")

def load_data(path=DEFAULT_PATH):
    if not os.path.exists(path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"students": []}
        save_data(data, path)
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or doesn't exist, return empty structure
        data = {"students": []}
        save_data(data, path)
        return data

def save_data(data, path=DEFAULT_PATH):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_student_result(student_id, topic, score):
    data = load_data()
    for s in data["students"]:
        if s["id"] == student_id:
            s.setdefault("history", []).append({"topic": topic, "score": score})
            save_data(data)
            return
    # if student not found, create
    new = {"id": student_id, "history": [{"topic": topic, "score": score}]}
    data["students"].append(new)
    save_data(data)

def get_student_data(student_id):
    """Get a specific student's data"""
    data = load_data()
    for s in data["students"]:
        if s["id"] == student_id:
            return s
    return None

def get_all_students():
    """Get all students data"""
    data = load_data()
    return data.get("students", [])