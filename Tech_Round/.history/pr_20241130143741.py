import os
from flask import Flask, render_template, request, redirect, url_for
import psycopg2
from psycopg2 import sql
from groq import Groq

# Initialize Flask app and Groq client with API key
app = Flask(__name__)

client = Groq(api_key="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")

# PostgreSQL connection parameters
DB_HOST = "localhost"
DB_NAME = "interview"
DB_USER = "postgres"
DB_PASSWORD = "22882288"

# Route to display the questions and collect answers
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        company_name = request.form.get("company_name")

        # Get all answers from the form
        answers = {}
        for i in range(1, 11):  # Expecting answers for 10 questions
            answer = request.form.get(f"answer_{i}")
            answers[i] = answer

        # Evaluate the answers and store in the database
        for i, answer in answers.items():
            question_text = QUESTIONS[i-1].format(company_name=company_name)  # Format the question
            feedback = evaluate_answer(answer, question_text)
            score = extract_score(feedback)
            
            store_answer_in_db(company_name, question_text, answer, feedback, score)
        
        return redirect(url_for('thank_you'))

    return render_template('HR_Round/index.html')

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
def store_answer_in_db(company_name, question, answer, feedback, score):
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
        )
        cursor = connection.cursor()

        # SQL query to insert the answer
        insert_query = sql.SQL("""
            INSERT INTO user_answers (company_name, question, answer, feedback, score)
            VALUES (%s, %s, %s, %s, %s)
        """)

        cursor.execute(insert_query, (company_name, question, answer, feedback, score))
        connection.commit()

        cursor.close()
        connection.close()

    except Exception as error:
        print(f"Error storing answer: {error}")

if __name__ == '__main__':
    app.run(debug=True)
