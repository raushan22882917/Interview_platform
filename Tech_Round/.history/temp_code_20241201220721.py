import os
from flask import Flask, request, jsonify, render_template, session
from groq import Groq

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # For session management

# Initialize Groq API client
client = Groq(api_key="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")

# Route for homepage (initial question)
@app.route('/')
def index():
    session.clear()  # Clear session for a new session
    return render_template('demo.html')

# Route to handle the first question and redirect to the steps page
@app.route('/start', methods=['POST'])
def start():
    session['question'] = request.form['question']  # Save the initial question
    session['current_step'] = "Example"  # Start from the first step
    return render_template('steps.html', step="Example", question=session['question'])

# Route to process each step dynamically
@app.route('/process_step', methods=['POST'])
def process_step():
    step = session['current_step']
    user_input = request.form['user_input']
    
    # Call the Groq API for this step
    prompt = f"Step: {step}\nUser Input: {user_input}\nQuestion: {session['question']}\nSolve the problem as per E.S.T.C.V methodology."
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )
    
    # Save the current step's result
    step_output = response.choices[0].message.content
    session[step] = {"input": user_input, "output": step_output}

    # Determine the next step
    steps = ["Example", "Solution", "Test Cases", "Code", "Validate"]
    current_index = steps.index(step)
    if current_index < len(steps) - 1:
        session['current_step'] = steps[current_index + 1]
        return jsonify({"next_step": session['current_step'], "step_output": step_output})
    else:
        session['current_step'] = None  # All steps are complete
        return jsonify({"final_step": True, "step_output": step_output})

# Route for final feedback
@app.route('/final_submit', methods=['POST'])
def final_submit():
    # Fetch all steps from session
    results = {step: session.get(step, {}) for step in ["Example", "Solution", "Test Cases", "Code", "Validate"]}
    
    # Generate feedback
    feedback = "All steps completed. Here's your feedback:\n"
    for step, data in results.items():
        feedback += f"Step {step}:\n- Input: {data.get('input', 'N/A')}\n- Output: {data.get('output', 'N/A')}\n\n"

    return jsonify({"feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True)
