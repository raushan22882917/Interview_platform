from flask import Flask, render_template, request, session, jsonify
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize the Groq API client
client = Groq(api_key="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")

# Define the steps in the E.S.T.C.V process
STEPS = ["Example", "Solution", "Test Cases", "Code", "Validation"]

@app.route('/')
def chatbot_interface():
    """
    Display the chatbot interface and initialize session variables.
    """
    session.clear()
    session['responses'] = {}
    return render_template('Ai/chatbot.html', steps=STEPS)

@app.route('/submit_question', methods=['POST'])
def submit_question():
    """
    Save the question to the session and return a success message.
    """
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Question is missing!"}), 400
    session['question'] = question
    return jsonify({"message": "Question submitted successfully!"})

@app.route('/submit_step', methods=['POST'])
def submit_step():
    """
    Process and save the user's response for a specific step.
    """
    step = request.json.get('step')
    user_input = request.json.get('input')

    if not step or not user_input:
        return jsonify({"error": "Step or input is missing!"}), 400

    feedback, score = process_step(step, user_input)
    session['responses'][step] = {'input': user_input, 'feedback': feedback, 'score': score}

    return jsonify({"feedback": feedback, "score": score})

@app.route('/final_submit', methods=['POST'])
def final_submit():
    """
    Aggregate all steps' responses, calculate the total score, and return the results.
    """
    if 'responses' not in session or not session['responses']:
        return jsonify({"error": "No responses to evaluate!"}), 400

    total_score = sum(response['score'] for response in session['responses'].values())
    return jsonify({"responses": session['responses'], "total_score": total_score})

def process_step(step, user_input):
    """
    Process the user input for a specific step and return feedback and score.
    """
    question = session.get('question', "No question provided.")
    feedback = ""
    score = 0  # Default score

    if step == "Example":
        inputs_outputs = user_input.split("\n")
        if len(inputs_outputs) < 3:
            feedback = "You need to provide at least 3 examples of input-output pairs."
            score = 2
        else:
            feedback = "Good job providing examples. Ensure they cover diverse scenarios."
            score = 5

    elif step == "Solution":
        prompt = f"Question: {question}\nUser's solution approach: {user_input}\nEvaluate the approach."
        feedback = call_groq_api(prompt)
        score = 4

    elif step == "Test Cases":
        test_cases = user_input.split("\n")
        if not test_cases:
            feedback = "Please provide at least one test case."
            score = 1
        elif "edge" not in user_input.lower():
            feedback = "You missed edge test cases. Consider edge conditions like minimum, maximum, or empty inputs."
            score = 3
        else:
            feedback = "Test cases are well thought out."
            score = 5

    elif step == "Code":
        prompt = f"Question: {question}\nCode: {user_input}\nEvaluate the correctness, time complexity, and space complexity."
        feedback = call_groq_api(prompt)
        score = 4

    elif step == "Validation":
        prompt = f"Question: {question}\nValidation of code: {user_input}\nCheck all test cases and analyze time and space complexity."
        feedback = call_groq_api(prompt)
        score = 5

    return feedback, score

def call_groq_api(prompt):
    """
    Call the Groq API with a given prompt and return the response.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error while calling Groq API: {e}"

if __name__ == '__main__':
    app.run(debug=True)
