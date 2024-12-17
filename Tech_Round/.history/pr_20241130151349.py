import os
from flask import Flask, render_template, request, jsonify
import psycopg2
from groq import Groq

# Initialize Flask app
app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key="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")

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",      # Database server
        database="interview",  # Database name
        user="postgres",  # Your database username
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    company_name = request.form['company_name']
    question_index = int(request.form['question_index'])
    user_answer = request.form['user_answer']
    
    questions = format_questions(company_name)
    question_text = questions[question_index]
    
    # Evaluate answer
    feedback = evaluate_answer(user_answer, question_text)
    
    # Score based on feedback (simplified)
    score = 2  # Default score (you can improve this logic)
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
    
    # Store in DB
    store_in_db(company_name, question_text, user_answer, feedback, score)
    
    # Return feedback and score to the UI
    return jsonify({
        'feedback': feedback,
        'score': score
    })

if __name__ == '__main__':
    app.run(debug=True)
