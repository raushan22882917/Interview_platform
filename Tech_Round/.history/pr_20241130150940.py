import os
import psycopg2
from groq import Groq

# Initialize Groq client with API key directly set
client = Groq(api_key="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")

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",      # Database server
        database="interview_db",  # Database name
        user="your_username",  # Your database username
        password="your_password" # Your database password
    )

# Function to format questions based on company name
def format_questions(company_name):
    return [
        f"Can you tell me about yourself?",
        f"Why do you want to work at {company_name}?",
        f"What are your greatest strengths?",
        f"Where do you see yourself in five years?",
        f"Why should we hire you over other candidates?",
        f"Describe a challenging situation you faced and how you resolved it.",
        f"What motivates you to perform well at work?",
        f"How do you handle criticism or feedback from your manager?",
        f"Can you share an example of how you worked effectively in a team?",
        f"What is your approach to time management and meeting deadlines?"
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

# Function to store data in the database
def store_in_db(company_name, question, user_answer, feedback, score):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    cursor.execute(
        """
        INSERT INTO interview_data (company_name, question, user_answer, feedback, score)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (company_name, question, user_answer, feedback, score)
    )
    
    connection.commit()
    cursor.close()
    connection.close()

# Function to simulate the chatbot experience
def chatbot():
    print("Hello! Please enter the name of your target company.")
    company_name = input("Enter company name: ")

    # Get the formatted questions for the specified company
    QUESTIONS = format_questions(company_name)

    print(f"\nGreat! Now I will ask you a series of questions, and then I'll evaluate your answers.\n")
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

        # Store the data in the database
        store_in_db(company_name, question, user_answer, feedback, score)
    
    # Show the total score after all questions
    print(f"Your total score: {total_score}/{len(QUESTIONS) * 5}")
    print("Thank you for your responses! Goodbye.")

# Run the chatbot
chatbot()
