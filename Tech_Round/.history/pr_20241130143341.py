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
DB_NAME = "your_db_name"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"

# Predefined questions based on the company selected
QUESTIONS_BY_COMPANY = {
    "TechCorp": [
        "Can you tell me about yourself?",
        "Why do you want to work at TechCorp?",
        "What are your greatest strengths?",
        "What motivates you to perform well at work?"
    ],
    "InnovateX": [
        "Why do you want to work at InnovateX?",
        "Describe a challenging situation you faced at InnovateX and how you resolved it.",
        "What are your greatest achievements?",
        "How do you handle criticism?"
    ],
    "HealthPlus": [
        "What attracts you to working at HealthPlus?",
        "What is your approach to healthcare industry challenges?",
        "How do you keep up with healthcare trends?",
        "Describe a time you worked with a healthcare team."
    ]
}

# Route to display the company selection form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the company selected by the user
        company = request.form.get('company')
        if company:
            questions = QUESTIONS_BY_COMPANY.get(company)
            return render_template('questions.html', company=company, questions=questions)
        else:
            return "Please select a company"

    return render_template('index.html')

# Route to display the thank you page after submitting answers
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
