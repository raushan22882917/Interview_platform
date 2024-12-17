from flask import Flask, render_template, request, session, redirect
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize the Groq API client
client = Groq(api_key="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")

# Define the steps in the E.S.T.C.V process
STEPS = ["Example", "Solution", "Test Cases", "Code", "Validation"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    session.clear()
    session['question'] = request.form['question']
    session['current_step'] = STEPS[0]
    session['responses'] = {}
    return redirect(f'/step?name={STEPS[0]}')

@app.route('/step', methods=['GET', 'POST'])
def step():
    step_name = request.args.get('name', STEPS[0])
    
    if request.method == 'POST':
        user_input = request.form['user_input']
        feedback = process_step(step_name, user_input)
        session['responses'][step_name] = {'input': user_input, 'feedback': feedback}

        # Determine the next step
        current_index = STEPS.index(step_name)
        if current_index < len(STEPS) - 1:
            next_step = STEPS[current_index + 1]
            return redirect(f'/step?name={next_step}')
        else:
            return redirect('/feedback')

    return render_template('step.html', step=step_name, question=session['question'], responses=session['responses'])

@app.route('/feedback')
def feedback():
    return render_template('/feedback.html', feedback=session['responses'])

def process_step(step, user_input):
    """Process the user input for a specific step and return feedback."""
    question = session['question']
    feedback = ""

    if step == "Example":
        inputs_outputs = user_input.split("\n")
        if len(inputs_outputs) < 3:
            feedback = "You need to provide at least 3 examples of input-output pairs."
        else:
            feedback = "Good job providing examples. Ensure they cover diverse scenarios."

    elif step == "Solution":
        prompt = f"Question: {question}\nUser's solution approach: {user_input}\nEvaluate the approach."
        feedback = call_groq_api(prompt)

    elif step == "Test Cases":
        test_cases = user_input.split("\n")
        if not test_cases:
            feedback = "Please provide at least one test case."
        elif "edge" not in user_input.lower():
            feedback = "You missed edge test cases. Consider edge conditions like minimum, maximum, or empty inputs."
        else:
            feedback = "Test cases are well thought out."

    elif step == "Code":
        prompt = f"Question: {question}\nCode: {user_input}\nEvaluate the correctness, time complexity, and space complexity."
        feedback = call_groq_api(prompt)

    elif step == "Validation":
        prompt = f"Question: {question}\nValidation of code: {user_input}\nCheck all test cases and analyze time and space complexity."
        feedback = call_groq_api(prompt)

    return feedback

def call_groq_api(prompt):
    """Call the Groq API with a given prompt."""
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
