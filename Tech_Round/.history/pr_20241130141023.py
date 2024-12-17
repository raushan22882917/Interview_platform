import requests

# Predefined questions
QUESTIONS = [
    "Can you tell me about yourself?",
    "Why do you want to work at our company?",
    "What are your greatest strengths?",
    "Where do you see yourself in five years?",
    "Why should we hire you over other candidates?",
    "Describe a challenging situation you faced and how you resolved it.",
    "What motivates you to perform well at work?",
    "How do you handle criticism or feedback from your manager?",
    "Can you share an example of how you worked effectively in a team?",
    "What is your approach to time management and meeting deadlines?"
]

# Function to evaluate user answer using Groq API
def evaluate_answer(user_answer, question_text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "prompt": (
            f"Evaluate the following answer: '{user_answer}' for the question: '{question_text}'. "
            f"Provide feedback and categorize it as one of these: Good, Bad, Very Good, Excellent, Awesome, or Need Improvement."
        )
    }
    
    response = requests.post(
        "https://api.groq.com/evaluate",
        headers=headers,
        json=payload
    )
    
    # Extract feedback and category
    data = response.json()
    feedback = data.get("feedback", "No feedback provided.")
    category = data.get("category", "Need Improvement")
    return feedback, category

# Function to process questions
def process_questions():
    for index, question in enumerate(QUESTIONS, start=1):
        # Simulate capturing user input (replace with actual input)
        user_answer = input(f"Answer for Question {index}: {question}\n> ")
        
        # Evaluate the answer
        feedback, category = evaluate_answer(user_answer, question)
        print(f"Feedback: {feedback}")
        print(f"Category: {category}\n")

# Groq API key
GROQ_API_KEY = "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"

# Run the question-answer process
process_questions()
