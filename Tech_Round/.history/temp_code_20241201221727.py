from flask import Flask, render_template, request, redirect, session, jsonify
import os
from groq import Groq

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize Groq API client
client = Groq(api_key="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")

# Define the E.S.T.C.V steps
STEPS = ["Example", "Solution", "Test Cases", "Code", "Validate"]

@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/start', methods=['POST'])
def start():
    session.clear()
    session['question'] = request.form['question']
    session['current_step'] = STEPS[0]
    return render_template('steps.html', step=STEPS[0], question=session['question'])

@app.route('/process_step', methods=['POST'])
def process_step():
    step = session['current_step']
    user_input = request.form['user_input']
    
    # Call the Groq API for the current step
    prompt = f"Step: {step}\nUser Input: {user_input}\nQuestion: {session['question']}\nSolve the problem as per E.S.T.C.V methodology."
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    # Save the output for the current step
    step_output = response.choices[0].message.content
    session[step] = {"input": user_input, "output": step_output}

    # Determine the next step
    current_index = STEPS.index(step)
    if current_index < len(STEPS) - 1:
        session['current_step'] = STEPS[current_index + 1]
        next_step = session['current_step']
        return render_template('steps.html', step=next_step, question=session['question'])
    else:
        session['current_step'] = None  # All steps are complete
        return render_template('final_feedback.html', feedback=session)

@app.route('/final_feedback', methods=['GET'])
def final_feedback():
    return render_template('final_feedback.html', feedback=session)

if __name__ == '__main__':
    app.run(debug=True)
