import os
from flask import Flask, render_template, request, redirect, url_for
import psycopg2
from psycopg2 import sql
from groq import Groq

# Initialize Flask app and Groq client with API key
app = Flask(__name__)

client = Groq(api_key="your_api_key_here")

# PostgreSQL connection parameters
DB_HOST = "localhost"
DB_NAME = "interview"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"

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

# Route to display the questions and collect answers
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get all answers from the form
        answers = {}
        for i, question in enumerate(QUESTIONS):
            answer = request.form.get(f"answer_{i+1}")
            answers[question] = answer

        # Evaluate the answers and store in the database
        for question, answer in answers.items():
            feedback = evaluate_answer(answer, question)
            score = extract_score(feedback)
            
            store_answer_in_db(question, answer, feedback, score)
        
        return redirect(url_for('thank_you'))

    return render_template('index.html', questions=QUESTIONS)

# Route for thank you page after submitting the answers
@app.route('/thank_you')
def thank_you():
    return "Thank you for your answers! They have been stored."

# Function to evaluate user answer using Groq API
def evaluate_answer(user_answer, question_text):
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
    feedback = chat_completion.choices[0].message.content
    return feedback

# Function to extract score from feedback
def extract_score(feedback):
    if "Bad" in feedback:
        return 1
    elif "Need Improvement" in feedback:
        return 2
    elif "Good" in feedback:
        return 3
    elif "Very Good" in feedback:
        return 4
    elif "Excellent" in feedback or "Awesome" in feedback:
        return 5
    else:
        return 2  # Default fallback score

# Function to store the answer in the PostgreSQL database
def store_answer_in_db(question, answer, feedback, score):
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
        )
        cursor = connection.cursor()

        # SQL query to insert the answer
        insert_query = sql.SQL("""
            INSERT INTO user_answers (question, answer, feedback, score)
            VALUES (%s, %s, %s, %s)
        """)

        cursor.execute(insert_query, (question, answer, feedback, score))
        connection.commit()

        cursor.close()
        connection.close()

    except Exception as error:
        print(f"Error storing answer: {error}")

if __name__ == '__main__':
    app.run(debug=True)
