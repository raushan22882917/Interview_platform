import os
from groq import Groq

# Initialize Groq client with API key from environment variable
client = Groq(api_key=os.environ.get("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"))

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
    "Excellent": 5,
    "Awesome": 5
}

# Function to evaluate user answer using Groq API
def evaluate_answer(user_answer, question_text):
    # Sending the question and user answer to Groq model for evaluation
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that evaluates answers to interview questions. "
                           "Provide feedback on the answer based on the following categories: "
                           "Good, Bad, Very Good, Excellent, Awesome, and Need Improvement. "
                           "Also, assign a score between 1 to 5 based on the answer quality."
            },
            {
                "role": "user",
                "content": f"Question: {question_text}\nAnswer: {user_answer}"
            }
        ],
        model="llama3-8b-8192",
    )

    # Extracting the response from the API
    feedback = chat_completion.choices[0].message.content
    return feedback

# Function to simulate the chatbot experience
def chatbot():
    print("Hello! I will ask you a series of questions, and then I'll evaluate your answers.")
    total_score = 0

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {index}: {question}")
        
        # Get user input for the question
        user_answer = input("Your answer: ")
        
        # Evaluate the answer using the LLM
        feedback = evaluate_answer(user_answer, question)
        
        # Print the feedback
        print("\nFeedback:")
        print(feedback)
        
        # Here, we assume a simple score extraction based on feedback (you can fine-tune this)
        if "Bad" in feedback:
            score = 1
        elif "Need Improvement" in feedback:
            score = 2
        elif "Good" in feedback:
            score = 3
        elif "Very Good" in feedback:
            score = 4
        elif "Excellent" in feedback or "Awesome" in feedback:
            score = 5
        else:
            score = 2  # Default fallback score

        total_score += score
        print(f"Score: {score}/5\n")
    
    # Show the total score after all questions
    print(f"Your total score: {total_score}/{len(QUESTIONS) * 5}")
    print("Thank you for your responses! Goodbye.")

# Run the chatbot
chatbot()
