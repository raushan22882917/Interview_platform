from flask import Flask, render_template, request, session, redirect
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize the Groq API client
client = Groq(api_key="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")

# Define the steps in the E.S.T.C.V process
STEPS = ["Example", "Solution", "Test Cases", "Code", "Validation"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Step Handling
        current_step = request.form.get('current_step')
        user_input = request.form.get('user_input')
        
        if 'responses' not in session:
            session['responses'] = {}
        
        feedback = process_step(current_step, user_input)
        
        # Store input and feedback for the current step
        session['responses'][current_step] = {'input': user_input, 'feedback': feedback}
        
        # Move to the next step or finish
        current_index = STEPS.index(current_step)
        if current_index < len(STEPS) - 1:
            next_step = STEPS[current_index + 1]
        else:
            next_step = None
        
        return render_template('Ai/editor.html', 
                               current_step=current_step, 
                               next_step=next_step, 
                               responses=session['responses'], 
                               question=session.get('question', ''))
    
    # Initialize the first step on the first visit
    return render_template('index.html', current_step=STEPS[0], next_step=STEPS[1], responses={}, question='')

def process_step(step, user_input):
    """Process the user input for a specific step and return feedback."""
    question = session.get('question', 'Your question here')
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
