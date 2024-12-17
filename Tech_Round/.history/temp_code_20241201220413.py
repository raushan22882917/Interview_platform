import os
from flask import Flask, request, jsonify, render_template, session
from groq import Groq

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # For session management

# Initialize Groq API client
client = Groq(api_key="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")

# Route for homepage
@app.route('/')
def index():
    # Clear session for new chat
    session.clear()
    return render_template('index.html')

# Route for handling each E.S.T.C.V step
@app.route('/process_step', methods=['POST'])
def process_step():
    step = request.form['step']
    user_input = request.form['user_input']

    # Call the Groq API for this step
    prompt = f"Step: {step}\nUser Input: {user_input}\nSolve the problem as per E.S.T.C.V methodology."
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )
    
    # Save the step's response in session
    step_output = response.choices[0].message.content
    session[step] = {"input": user_input, "output": step_output}

    return jsonify({"step_output": step_output})

# Route for final submission
@app.route('/final_submit', methods=['POST'])
def final_submit():
    # Fetch all steps from the session
    results = {step: session.get(step, {}) for step in ["Example", "Solution", "Test Cases", "Code", "Validate"]}
    
    # Generate feedback
    feedback = "All steps completed. Here's your feedback:\n"
    for step, data in results.items():
        feedback += f"Step {step}:\n- Input: {data.get('input', 'N/A')}\n- Output: {data.get('output', 'N/A')}\n\n"

    return jsonify({"feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True)
