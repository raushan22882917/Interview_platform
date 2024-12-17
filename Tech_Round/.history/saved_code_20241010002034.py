from flask import Flask, request, jsonify, render_template
import subprocess
import os
import psycopg2
import json

app = Flask(__name__)

# Database connection details
DB_HOST = 'localhost'
DB_NAME = 'interview'
DB_USER = 'postgres'
DB_PASS = '22882288'

# Connect to PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

# Sample questions (define these appropriately)
# Sample questions (define these appropriately)
questions = [
    {
        "title": "Addition Function",
        "question": "Write a function to add two numbers.",
        "test_cases": [
            {"input": "2 3\n", "expected_output": "5\n"},
            {"input": "10 15\n", "expected_output": "25\n"},
        ]
    },
    {
        "title": "Multiplication Function",
        "question": "Write a function to multiply two numbers.",
        "test_cases": [
            {"input": "2 3\n", "expected_output": "6\n"},
            {"input": "4 5\n", "expected_output": "20\n"},
        ]
    },
    {
        "title": "Even Number Check",
        "question": "Check if a number is even.",
        "test_cases": [
            {"input": "4\n", "expected_output": "True\n"},
            {"input": "5\n", "expected_output": "False\n"},
        ]
    },
    {
        "title": "String Reversal",
        "question": "Reverse a string.",
        "test_cases": [
            {"input": "abcd\n", "expected_output": "dcba\n"},
            {"input": "hello\n", "expected_output": "olleh\n"},
        ]
    },
    {
        "title": "Factorial Calculation",
        "question": "Calculate factorial of a number.",
        "test_cases": [
            {"input": "5\n", "expected_output": "120\n"},
            {"input": "3\n", "expected_output": "6\n"},
        ]
    },
]


@app.route('/editor')
def editor():
    return render_template('editor.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    code = data['code']
    language = data['language']
    class_name = data['class_name']  # Get the class name from the request
    question_id = int(data['question_id'])  # Convert to integer

    question = questions[question_id]
    test_cases = question["test_cases"]

    results = []
    all_passed = True  # Track if all test cases passed
    passed_count = 0  # Counter for passed test cases

    # Write the user code to a temporary file based on the language
    if language == 'python':
        filename = 'user_code.py'
        with open(filename, 'w') as f:
            f.write(code)
        command = ['python3', filename]
    elif language == 'cpp':
        filename = 'user_code.cpp'
        with open(filename, 'w') as f:
            f.write(code)
        compile_command = ['g++', filename, '-o', 'user_code.exe']  # Use .exe for Windows
        subprocess.run(compile_command)  # Compile the code
        command = ['./user_code.exe']  # Run the compiled program
    elif language == 'java':
        filename = f'{class_name}.java'  # Use the class name specified by the user
        with open(filename, 'w') as f:
            f.write(code)
        compile_command = ['javac', filename]  # Compile the Java code
        compile_result = subprocess.run(compile_command, capture_output=True, text=True)  # Compile the code

        # Check for compilation errors
        if compile_result.returncode != 0:
            results.append({"output": compile_result.stderr.strip(), "is_correct": False})
            all_passed = False
            save_results_to_db(question_id, code, language, results, all_passed, 0)
            return jsonify({"results": results, "all_passed": all_passed, "message": "Some test cases failed."})

        command = ['java', '-cp', '.', class_name]  # Run the compiled Java program with classpath set to current directory

    try:
        for test_case in test_cases:
            input_data = test_case["input"]
            expected_output = test_case["expected_output"]

            # Execute the code for each test case
            result = subprocess.run(command, input=input_data, text=True, capture_output=True)
            output = result.stdout.strip()
            error = result.stderr.strip()

            if error:
                results.append({"output": error, "is_correct": False})
                all_passed = False  # At least one test case failed
            else:
                is_correct = output == expected_output.strip()
                results.append({"output": output, "is_correct": is_correct})
                if is_correct:
                    passed_count += 1  # Increment the passed test case count
                else:
                    all_passed = False  # At least one test case failed

        # Save results to database, including passed test cases count
        save_results_to_db(question_id, code, language, results, all_passed, passed_count)

    finally:
        # Clean up the generated files
        if os.path.exists(filename):
            os.remove(filename)
        if language == 'cpp' and os.path.exists('user_code.exe'):
            os.remove('user_code.exe')
        if language == 'java' and os.path.exists(f'{class_name}.class'):
            os.remove(f'{class_name}.class')

    return jsonify({
        "message": "All test cases passed!" if all_passed else "Some test cases failed.",
        "passed_count": passed_count 
    })

# Update the function to save results to the database
def save_results_to_db(question_id, code, language, results, all_passed, passed_count):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO evaluation_results (question_id, code, language, results, all_passed, passed_test_cases_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (question_id, code, language, json.dumps(results), all_passed, passed_count)  # Add passed_count
        )
        conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    app.run(debug=True)
