from flask import Flask, request, render_template, session, redirect, url_for, flash
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from groq import Groq

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', "77b155a7046fa7ec6da9cabd7a8475e123d632df9e58c32f")  # Use environment variable

# Initialize the GROQ client with the API key
api_key = "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"  # Set your API key directly for testing (not recommended for production)
client = Groq(api_key=api_key)

# Load the questions from the CSV file
csv_file_name = 'ml.csv'
questions_df = pd.read_csv(f'course_data/{csv_file_name}', on_bad_lines='skip')
questions = questions_df['Question'].tolist()
answers = questions_df['Answer'].tolist()

# Load the SBERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route('/', methods=['GET', 'POST'])
def Tech():
    # Initialize session variables
    if 'question_index' not in session:
        session['question_index'] = 0
        session['user_answers'] = []

    # If all questions are attempted, redirect user
    if session['question_index'] >= len(questions):
        flash("You have attempted all questions. Redirecting to the start.", "info")
        
        # Reset session for a new attempt
        session.pop('question_index', None)
        session.pop('user_answers', None)

        return redirect(url_for('Tech'))  # Redirect to start over

    if request.method == 'POST':
        user_answer = request.form.get('answer', '')
        question_index = session['question_index']
        
        if question_index < len(questions):
            current_question = questions[question_index]
            correct_answer = answers[question_index]
            similarity_score = compute_similarity(user_answer, correct_answer)

            # Append user's answer to session
            session['user_answers'].append(user_answer)

            # Provide feedback based on similarity score
            feedback = ""
            ai_answer = ""
            user_feedback = ""
            suggested_words = ""

            if similarity_score >= 0.8:
                feedback = "Great job! Your answer is very similar to the correct one."
                ai_answer = generate_answer(current_question)  # Generate AI answer for the question
                user_feedback = f"Your answer: {user_answer}<br>Correct answer: {correct_answer}<br>AI-generated answer: {ai_answer}"
                session['question_index'] += 1  # Move to the next question

            elif similarity_score > 0.5:
                feedback = "Good attempt! Your answer is somewhat similar. Here’s some grammar feedback."
                ai_answer = generate_answer(current_question)  # Generate AI answer for the question
                suggested_words = suggest_better_words(user_answer)  # Get grammar feedback and suggestions
                user_feedback = f"Your answer: {user_answer}<br>Correct answer: {correct_answer}<br>AI-generated answer: {ai_answer}"
                session['question_index'] += 1  # Move to the next question
                
            else:
                feedback = "Your answer was quite different from the expected one. Generating an AI answer for you."
                ai_answer = generate_answer(current_question)  # Generate AI answer for the question
                suggested_words = suggest_better_words(user_answer)  # Suggest better wording based on user's answer
                user_feedback = f"Your answer: {user_answer}<br>Correct answer: {correct_answer}<br>AI-generated answer: {ai_answer}"

            return render_template('question.html', 
                                   question=questions[session['question_index']] if session['question_index'] < len(questions) else "All questions completed.",
                                   question_index=session['question_index'],
                                   total_questions=len(questions),
                                   user_answer=user_answer,
                                   correct_answer=correct_answer,
                                   similarity_score=similarity_score,
                                   feedback=feedback,
                                   user_feedback=user_feedback,
                                   suggested_words=suggested_words)  # Pass the suggested words to the template

    # Render the question for the user
    return render_template('question.html', 
                           question=questions[session['question_index']],
                           question_index=session['question_index'], 
                           total_questions=len(questions),
                           feedback=None)

def compute_similarity(user_answer, correct_answer):
    embeddings = model.encode([user_answer, correct_answer])
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(similarity)

def generate_answer(user_query):
    """Generate an answer to the user's query using the GROQ API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

def suggest_better_words(user_answer):
    """Suggest better words for the user's answer using the GROQ API."""
    # Example call to the GROQ API for grammar checking and suggesting better words
    grammar_feedback = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Check this answer for grammar mistakes and suggest better wording: '{user_answer}'",
            }
        ],
        model="llama3-8b-8192",
    )
    return grammar_feedback.choices[0].message.content

if __name__ == "__main__":
    app.run(debug=True)
