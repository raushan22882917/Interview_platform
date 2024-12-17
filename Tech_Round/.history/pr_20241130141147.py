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

# Scoring based on categories
CATEGORY_SCORES = {
    "Bad": 1,
    "Need Improvement": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 4,
    "Awesome": 5
}

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
            f"Provide feedback, categorize it as Good, Bad, Very Good, Excellent, Awesome, or Need Improvement, and give a score from 1 to 5."
        )
    }
    
    response = requests.post(
        "https://api.groq.com/evaluate",
        headers=headers,
        json=payload
    )
    
    data = response.json()
    feedback = data.get("feedback", "No feedback provided.")
    category = data.get("category", "Need Improvement")
    score = CATEGORY_SCORES.get(category, 2)
    
    return feedback, category, score

# Function to process questions
def process_questions():
    total_score = 0
    for index, question in enumerate(QUESTIONS, start=1):
        # Simulate capturing user input (replace with actual input)
        user_answer = input(f"Answer for Question {index}: {question}\n> ")
        
        # Evaluate the answer
        feedback, category, score = evaluate_answer(user_answer, question)
        total_score += score
        
        # Display results
        print(f"Feedback: {feedback}")
        print(f"Category: {category}")
        print(f"Score: {score}/5\n")
    
    print(f"Total Score: {total_score}/{len(QUESTIONS) * 5}")

# Groq API key
GROQ_API_KEY = "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"

# Run the question-answer process
process_questions()
