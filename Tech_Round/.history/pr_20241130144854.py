import os
from flask import Flask, render_template, request, redirect, url_for, session
import psycopg2
from psycopg2 import sql
from groq import Groq

# Initialize Flask app and Groq client with API key
app = Flask(__name__)
app.secret_key = 'sdfgrthujhyjgtfrdefsfghjbn'  # Needed for session management

client = Groq(api_key="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")

# PostgreSQL connection parameters
DB_HOST = "localhost"
DB_NAME = "interview"
DB_USER = "postgres"
DB_PASSWORD = "22882288"

# Predefined interview questions
QUESTIONS = [
    "Can you tell me about yourself and how your experience aligns with the values and culture of {}?",
    "Why do you want to work at {} and what specifically excites you about the company's mission and vision?",
    "What are your greatest strengths, and how do you think they will benefit you in your role at {}?",
    "Where do you see yourself in five years, and how does {} fit into your long-term career goals?",
    "Why should we hire you over other candidates, and how do your skills uniquely align with {}'s needs and goals?",
    "Describe a challenging situation you faced and how you resolved it. How would you handle a similar challenge in the context of {}'s work environment?",
    "What motivates you to perform well at work, and how can that motivation contribute to {}'s success?",
    "How do you handle criticism or feedback from your manager, and how would you respond to feedback at {}?",
    "Can you share an example of how you worked effectively in a team? How would you contribute to team dynamics at {}?",
    "What is your approach to time management and meeting deadlines, and how do you think that would be an asset at {}?"
]

# Route to display the questions and collect answers
@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize session variables if not already set
    if 'company_name' not in session:
        session['company_name'] = None
    if 'current_question' not in session:
        session['current_question'] = 0
    if 'answers' not in session:
        session['answers'] = []

    if request.method == 'POST':
        # If it's the first form submission, set the company name
        if session['company_name'] is None:
            session['company_name'] = request.form['company_name']
        
        # If it's an answer submission, store the answer
        if session['company_name'] is not None:
            answer = request.form['answer']
            session['answers'].append(answer)

            # Evaluate and store the answer after all questions are answered
            if len(session['answers']) == 10:
                for idx, answer in enumerate(session['answers']):
                    question_text = QUESTIONS[idx].format(session['company_name'])
                    feedback = evaluate_answer(answer, question_text)
                    score = extract_score(feedback)
                    store_answer_in_db(session['company_name'], question_text, answer, feedback, score)
                
                # Reset for new session after answers are submitted
                return redirect(url_for('thank_you'))

            # Increment to show the next question
            session['current_question'] += 1

    # Show the current question and answers
    return render_template('H/index.html', company_name=session['company_name'], 
                           current_question=session['current_question'], questions=QUESTIONS)

# Route for thank you page after submitting the answers
@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

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
