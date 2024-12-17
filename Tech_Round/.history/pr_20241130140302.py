import pandas as pd
import requests

# Function to evaluate user answer using Groq API
def evaluate_answer(user_answer, question_text):
    # Send user answer and question to Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.groq.com/evaluate",
        headers=headers,
        json={
            "question": question_text,
            "user_answer": user_answer
        }
    )
    
    # Parse Groq's response
    groq_feedback = response.json().get("feedback")
    return groq_feedback

# Read questions from CSV file
def process_questions(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Iterate through each question
    for index, row in df.iterrows():
        question_id = row["QuestionID"]
        question_text = row["Question"]
        
        # Simulate capturing user answer (replace with actual user input)
        user_answer = input(f"Answer for Question {question_id}: {question_text}\n> ")
        
        # Evaluate the answer
        feedback = evaluate_answer(user_answer, question_text)
        print(f"Feedback for Question {question_id}: {feedback}\n")

# Path to your CSV file
file_path = "Hr
hr_round.csv"

# Groq API key
GROQ_API_KEY = "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"

# Process questions
process_questions(file_path)
