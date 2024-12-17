from flask import Flask, render_template, request, jsonify, session
import psycopg2
import random
import requests
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure random key

# Database configuration
DB_CONFIG = {
    "dbname": "interview",
    "user": "postgres",
    "password": "22882288",
    "host": "localhost",
    "port": "5432",
}

# GroQ API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Replace with your GroQ API Key


def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(**DB_CONFIG)


def reframe_questions(company, questions):
    """Reframe questions based on the selected company using GroQ API."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    reframed_questions = []

    for question in questions:
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"Reframe the following question for a company like {company}."},
                {"role": "user", "content": question}
            ]
        }
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            reframed_question = result.get("choices", [{}])[0].get("message", {}).get("content", question)
            reframed_questions.append(reframed_question)
        else:
            reframed_questions.append(question)  # Fallback to original question if API fails
    return reframed_questions


@app.route("/hr_index")
def hr_index():
    """Home page for selecting a company."""
    companies = ["Google", "Amazon", "Microsoft", "Facebook", "Apple"]
    return render_template("HR_Round/index.html", companies=companies)


@app.route("/start_hr_round", methods=["POST"])
def start_hr_round():
    """Start the HR round by selecting a company."""
    company = request.form.get("company")
    if not company:
        return jsonify({"error": "Company not selected."}), 400

    session["company"] = company

    # Example set of HR questions
    original_questions = [
        "Tell me about yourself.",
        "Why do you want to work at our company?",
        "What are your strengths?",
        "What are your weaknesses?",
        "Describe a challenging situation you handled.",
        "How do you handle conflict?",
        "Where do you see yourself in five years?",
        "Why should we hire you?",
        "What motivates you?",
        "Describe your leadership style."
    ]

    # Reframe questions based on the company
    session["questions"] = reframe_questions(company, original_questions)
    session["current_question"] = 0
    session["answers"] = []

    return jsonify({"redirect_url": "/questions"})


@app.route("/questions", methods=["GET"])
def questions_page():
    """Serve the questions page."""
    if "company" not in session or "questions" not in session:
        return jsonify({"error": "HR round not started."}), 400
    return render_template("HR_Round/result.html")


@app.route("/next_question", methods=["POST"])
def next_question():
    """Provide the next question and save the previous answer."""
    if "current_question" not in session or session["current_question"] >= len(session["questions"]):
        return jsonify({"status": "finished"})

    # Save previous answer
    if "answer" in request.json:
        answer = request.json["answer"]
        question = session["questions"][session["current_question"]]
        score = 0  # Score evaluation can be added here if needed

        # Save to database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO hrround (company, question, answer, score, date) VALUES (%s, %s, %s, %s, %s)",
            (session["company"], question, answer, score, datetime.now()),
        )
        conn.commit()
        cur.close()
        conn.close()

        session["answers"].append({"question": question, "answer": answer, "score": score})
        session["current_question"] += 1

    # Get the next question
    if session["current_question"] < len(session["questions"]):
        return jsonify({"question": session["questions"][session["current_question"]]})
    else:
        return jsonify({"status": "finished"})


@app.route("/hr_results", methods=["GET"])
def hr_results():
    """Show the results after the HR round."""
    return jsonify({"answers": session.get("answers", [])})


if __name__ == "__main__":
    app.run(debug=True)
