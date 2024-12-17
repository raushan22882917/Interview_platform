from flask import Flask, render_template, request, session, redirect
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sdfghjukl;kjhugfdsfghjk")  # For security, consider using an environment variable

# Initialize the Groq API client
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "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"))  # Get the Groq API key from environment variables

# Define the steps in the E.S.T.C.V process
STEPS = ["Example", "Solution", "Test Cases", "Code", "Validation"]

@app.route('/')
def index():
    return render_template('Ai/index.html')

@app.route('/start', methods=['POST'])
def start():
    session.clear()
    session['question'] = request.form['question']
    session['current_step'] = STEPS[0]
    session['responses'] = {}
    session['marks'] = {}  # Store marks for each step
    return redirect(f'/step')

@app.route('/step', methods=['GET', 'POST'])
def step():
    current_step = session.get('current_step', STEPS[0])
    responses = session.get('responses', {})

    if request.method == 'POST':
        user_input = request.form['user_input']
        feedback, score = process_step(current_step, user_input)
        responses[current_step] = {'input': user_input, 'feedback': feedback, 'score': score}

        session['responses'] = responses
        session['marks'][current_step] = score  # Store score for the current step

        # Move to next step
        current_index = STEPS.index(current_step)
        if current_index < len(STEPS) - 1:
            session['current_step'] = STEPS[current_index + 1]
            return redirect(f'/step')
        else:
            total_score = sum(session['marks'].values())  # Calculate total score
            return render_template('Ai/feedback.html', responses=responses, total_score=total_score)

    return render_template('Ai/steps.html', step=current_step, question=session['question'], responses=responses)

def process_step(step, user_input):
    """Process the user input for a specific step and return feedback and score."""
    question = session['question']
    feedback = ""
    score = 0  # Default score

    if step == "Example":
        inputs_outputs = user_input.split("\n")
        if len(inputs_outputs) < 3:
            feedback = "You need to provide at least 3 examples of input-output pairs."
            score = 2  # Lower score for insufficient examples
        else:
            feedback = "Good job providing examples. Ensure they cover diverse scenarios."
            score = 5  # Full score for good examples

    elif step == "Solution":
        prompt = f"Question: {question}\nUser's solution approach: {user_input}\nEvaluate the approach."
        feedback = call_groq_api(prompt)
        score = 4  # Placeholder, adjust based on feedback evaluation

    elif step == "Test Cases":
        test_cases = user_input.split("\n")
        if not test_cases:
            feedback = "Please provide at least one test case."
            score = 1  # Low score for no test cases
        elif "edge" not in user_input.lower():
            feedback = "You missed edge test cases. Consider edge conditions like minimum, maximum, or empty inputs."
            score = 3  # Medium score for missing edge cases
        else:
            feedback = "Test cases are well thought out."
            score = 5  # Full score for good test cases

    elif step == "Code":
        prompt = f"Question: {question}\nCode: {user_input}\nEvaluate the correctness, time complexity, and space complexity."
        feedback = call_groq_api(prompt)
        score = 4  # Placeholder for code quality evaluation

    elif step == "Validation":
        prompt = f"Question: {question}\nValidation of code: {user_input}\nCheck all test cases and analyze time and space complexity."
        feedback = call_groq_api(prompt)
        score = 5  # High score for complete validation

    return feedback, score

def call_groq_api(prompt):
    """Call the Groq API with a given prompt."""
    try:
        response = client.chat.completions.create(
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",  # You can modify the model if necessary
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error while calling Groq API: {e}"

if __name__ == '__main__':
    app.run(debug=True)
