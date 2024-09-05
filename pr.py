from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

# Define questions and test cases
QUESTIONS = {
    "1": {
        "question": "Write a function to return the sum of two integers.",
        "test_cases": [
            {"input": "2, 3", "expected_output": "5"},
            {"input": "10, 15", "expected_output": "25"}
        ]
    },
    "2": {
        "question": "Write a function to find the maximum of three integers.",
        "test_cases": [
            {"input": "1, 2, 3", "expected_output": "3"},
            {"input": "7, 5, 6", "expected_output": "7"}
        ]
    }
}

@app.route('/startcoding')
def startcoding():
    return render_template('HR_Round/index.html', questions=QUESTIONS)

@app.route('/run_code', methods=['POST'])
def run_code():
    code = request.form.get('code')
    question_id = request.form.get('question_id')
    question_data = QUESTIONS.get(question_id, {})
    test_cases = question_data.get('test_cases', [])
    
    # Save code to a file
    filename = 'temp_code.py'
    with open(filename, 'w') as f:
        f.write(code)
    
    # Function to run code and get output
    def get_output(input_data):
        try:
            result = subprocess.check_output(
                ['python', filename],
                input=input_data,
                text=True,  # Ensure input is treated as string
                timeout=5
            )
            return result.strip()
        except subprocess.CalledProcessError as e:
            return str(e)
        except subprocess.TimeoutExpired:
            return "Code execution timed out"

    # Run code against all test cases
    results = []
    all_passed = True
    for test_case in test_cases:
        input_data = test_case['input']
        expected_output = test_case['expected_output']
        output = get_output(input_data)
        if output != expected_output:
            all_passed = False
        results.append({
            "input": input_data,
            "expected_output": expected_output,
            "actual_output": output
        })

    if all_passed:
        return jsonify({"message": "Congratulations! All test cases passed!"})
    else:
        return jsonify({"results": results})

@app.route('/questions')
def questions():
    return jsonify(QUESTIONS)

if __name__ == '__main__':
    app.run(debug=True)
