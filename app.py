from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
import psycopg2
import smtplib
import requests
import subprocess
import tempfile
import os
import json
from email.mime.text import MIMEText
from datetime import datetime
import random
from bs4 import BeautifulSoup
import math
import csv
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from psycopg2 import sql
import pickle
import difflib
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import language_tool_python
import json

app = Flask(__name__)
app.secret_key = 'wertyuytrewaawsdesfrdgtfhytjdfghjk'  # Change this to a random secret key

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="interview",
        user="postgres",  # Replace with your PostgreSQL username
        password="22882288"  # Replace with your PostgreSQL password
    )
    return conn

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Store in PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                        (name, email, password))
            conn.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except psycopg2.IntegrityError:
            conn.rollback()
            flash('Email already exists. Please try another.', 'danger')
        finally:
            cur.close()
            conn.close()

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            session['user_email'] = user[2]
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

# Dashboard route (after login)
@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        return render_template('dashboard.html', user_name=session['user_name'])
    else:
        return redirect(url_for('login'))

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Index route
@app.route('/')
def index():
    # Load the data from the CSV file
    df = pd.read_csv('data/article_data.csv')
    
    # Convert the dataframe to a list of dictionaries
    data = df.to_dict(orient='records')
    
    # Pass the data to the index.html template
    return render_template('index.html', data=data)

# Topics route
@app.route('/topics')
def topics():
    if 'user_id' in session:
        return render_template('/techskill/subject/topic.html')
    else:
        return redirect(url_for('login'))



################################################################################################################################
# Database configuration
DB_HOST = 'localhost'
DB_NAME = 'interview'
DB_USER = 'postgres'
DB_PASSWORD = '22882288'

# Email configuration
EMAIL_HOST = 'smtp.gmail.com'  # Replace with your SMTP server
EMAIL_PORT = 587
EMAIL_USERNAME = 'raushan2288.jnvbanka@gmail.com'  # Replace with your email
EMAIL_PASSWORD = 'dqooryackjjjwdzp'  # Replace with your email password



@app.route('/instruction')
def instructions():
    if 'user_id' in session:
        return render_template('video_int/home.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/instructiondetail')
def int():
    if 'user_id' in session:
        return render_template('video_int/inst.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/About')
def about():
    if 'user_id' in session:
        return render_template('about.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/hrinst')
def hrinst():
    if 'user_id' in session:
        return render_template('HR_Round/instruction.html')
    else:
        return redirect(url_for('login'))

def generate_unique_number():
    return f"{random.randint(100000, 999999)}"

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

@app.route('/book_slot', methods=['GET', 'POST'])
def book_slot():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        date = request.form['date']
        time = request.form['time']
        unique_number = generate_unique_number()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO bookings (name, email, date, time, unique_number) VALUES (%s, %s, %s, %s, %s)",
            (name, email, date, time, unique_number)
        )
        conn.commit()
        cur.close()
        conn.close()

        # Send email notification
        send_email_notification(email, date, time, unique_number)

        flash('Slot booked successfully!', 'success')
        return redirect(url_for('book_slot'))

    return render_template('video_int/booking.html')

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_notification(to_email, date, time, unique_number):
    subject = '🎉 Your TechPrep Video Interview Slot is Confirmed!'

    html_body = f'''
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                line-height: 1.6;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 30px auto;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            h2 {{
                color: #4CAF50;
            }}
            p {{
                font-size: 16px;
            }}
            .highlight {{
                font-weight: bold;
                color: #1E88E5;
            }}
            .footer {{
                margin-top: 20px;
                text-align: center;
                font-size: 14px;
                color: #888;
            }}
            .footer a {{
                color: #1E88E5;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎉 Your TechPrep Video Interview Slot is Confirmed!</h2>
            <p>Dear Candidate,</p>
            <p>We are pleased to inform you that your video interview has been successfully scheduled through <span class="highlight">TechPrep</span>!</p>
            <p>🗓 <span class="highlight">Date:</span> {date}</p>
            <p>🕒 <span class="highlight">Time:</span> {time}</p>
            <p>🔢 <span class="highlight">Booking Number:</span> {unique_number}</p>
            <p>Please keep this booking number handy for reference. We look forward to having you in the interview process.</p>
            <div class="footer">
                Best Regards,<br>
                <span class="highlight">TechPrep Team</span><br>
                <em>Your Gateway to Success</em><br>
                <a href="https://www.techprep.com">Visit TechPrep</a>
            </div>
        </div>
    </body>
    </html>
    '''

    # Create the email message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = EMAIL_USERNAME
    msg['To'] = to_email

    # Attach the HTML body to the email
    msg.attach(MIMEText(html_body, 'html'))

    # Send the email
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)

        
        
@app.route('/join_interview', methods=['GET', 'POST'])
def join_interview():
    if request.method == 'POST':
        unique_number = request.form['unique_number']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, email FROM bookings WHERE unique_number = %s", (unique_number,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user:
            name, email = user
            return render_template('video_int/inst.html', name=name, email=email)
        else:
            flash('Invalid unique number. Please try again.', 'danger')
            return redirect(url_for('instructions'))

    return redirect(url_for('instructions'))

# ################################################################################################editor
# def load_questions():
#     # Adjust the path to where your questions.json file is located
#     file_path = os.path.join('static', 'Question/questions.json')
#     with open(file_path, 'r') as f:
#         return json.load(f)

# # questions = load_questions()


@app.route('/questions')
def question_page():
    return render_template('checks/questions.html')

# @app.route('/solve/<int:question_id>')
# def solve_question(question_id):
#     question = next((q for q in questions if q['id'] == question_id), None)
#     if question is None:
#         return "Question not found", 404
#     return render_template('checks/editor.html', question=question)

# @app.route('/run_code', methods=['POST'])
# def run_code():
#     code = request.json.get('code', '')
#     question_id = request.json.get('question_id')
#     language = request.json.get('language', 'python')

#     question = next((q for q in questions if q['id'] == question_id), None)
#     if question is None:
#         return jsonify({"result": "Question not found"}), 404

#     results = []
#     for i, test_case in enumerate(question['test_cases'], start=1):
#         input_values = test_case['input']
#         expected_output = test_case['expected_output']
        
#         try:
#             if language == 'python':
#                 exec_globals = {'input_values': input_values, 'output': None}
#                 exec(code, exec_globals)
#                 result = str(exec_globals.get('output', 'No output'))
            
#             elif language == 'java':
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.java') as java_file:
#                     java_file.write(code.encode())
#                     java_file.close()

#                     # Compile the Java code
#                     compile_process = subprocess.run(['javac', java_file.name], check=True, stderr=subprocess.PIPE)
#                     if compile_process.returncode != 0:
#                         return f"Compilation error: {compile_process.stderr.decode()}"

#                     # Run the compiled Java class
#                     classpath = os.path.dirname(java_file.name)
#                     result_process = subprocess.run(
#                         ['java', '-cp', classpath, 'Solution'],
#                         input=input_values,
#                         text=True,
#                         capture_output=True
#                     )
#                     result = result_process.stdout.strip()

#                     # Clean up temporary files
#                     os.remove(java_file.name)
#                     os.remove(java_file.name.replace('.java', '.class'))

            
#             elif language == 'c++':
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.cpp') as cpp_file:
#                     cpp_file.write(code.encode())
#                     cpp_file.close()
#                     # Compile the C++ code
#                     compile_process = subprocess.run(
#                         ['g++', cpp_file.name, '-o', 'Solution'],
#                         check=True, stderr=subprocess.PIPE
#                     )
#                     if compile_process.returncode != 0:
#                         return jsonify({"result": f"Compilation error: {compile_process.stderr.decode()}"}), 400

#                     # Run the compiled C++ program
#                     result_process = subprocess.run(
#                         ['./Solution'],
#                         input=str(input_values),
#                         text=True,
#                         capture_output=True
#                     )
#                     result = result_process.stdout.strip()
#                     os.remove(cpp_file.name)
#                     os.remove('Solution')
            
#             elif language == 'c':
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.c') as c_file:
#                     c_file.write(code.encode())
#                     c_file.close()
#                     # Compile the C code
#                     compile_process = subprocess.run(
#                         ['gcc', c_file.name, '-o', 'Solution'],
#                         check=True, stderr=subprocess.PIPE
#                     )
#                     if compile_process.returncode != 0:
#                         return jsonify({"result": f"Compilation error: {compile_process.stderr.decode()}"}), 400

#                     # Run the compiled C program
#                     result_process = subprocess.run(
#                         ['./Solution'],
#                         input=str(input_values),
#                         text=True,
#                         capture_output=True
#                     )
#                     result = result_process.stdout.strip()
#                     os.remove(c_file.name)
#                     os.remove('Solution')
            
#             else:
#                 return jsonify({"result": "Unsupported language"}), 400

#             is_passed = result == expected_output
#         except subprocess.CalledProcessError as e:
#             result = str(e.stderr.decode())
#             is_passed = False
#         except Exception as e:
#             result = str(e)
#             is_passed = False

#         results.append({
#             "test_case": i,
#             "input": input_values,
#             "expected_output": expected_output,
#             "result": result,
#             "passed": is_passed
#         })

#     return jsonify({"results": results})


########################################################################Blog Post Route
def load_data(filename):
    data = []
    try:
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data

def safe_int(value, default=1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# Route for the index page with category filter and pagination
@app.route('/Blog')
def blog():
    # Load data
    scraped_data = load_data('data/scraped_data.csv')

    # Get the selected category from query parameters (if any)
    selected_category = request.args.get('category', 'All')

    # Filter data by category if selected
    if selected_category != 'All':
        scraped_data = [row for row in scraped_data if row['Category'].lower() == selected_category.lower()]

    # Pagination
    page = safe_int(request.args.get('page', 1))
    per_page = 6
    total = len(scraped_data)
    pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = scraped_data[start:end]

    # Get unique categories for filtering
    all_data = load_data('data/scraped_data.csv')
    categories = sorted(set(row['Category'] for row in all_data))

    return render_template(
        'Blog/index.html',
        data=paginated_data,
        page=page,
        pages=pages,
        selected_category=selected_category,
        categories=categories
    )

# Route for the content page
@app.route('/content')
def content():
    link = request.args.get('link')
    if not link:
        return "Link parameter is missing", 400
    
    scraped_content = load_data('data/scraped_content.csv')
    
    # Find the corresponding content by link
    content_data = next((item for item in scraped_content if item['Link'] == link), None)
    
    if content_data is None:
        return "Content not found", 404

    return render_template('Blog/content.html', content=content_data)
########################################################################################
DATABASE_CONFIG = {
    'dbname': 'interview',
    'user': 'postgres',
    'password': '22882288',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    conn = psycopg2.connect(**DATABASE_CONFIG)
    return conn

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    
    # Save to database
    conn = get_db_connection()
    cur = conn.cursor()
    insert_query = sql.SQL("""
        INSERT INTO contact_messages (name, email, message)
        VALUES (%s, %s, %s)
    """)
    cur.execute(insert_query, (name, email, message))
    conn.commit()
    cur.close()
    conn.close()
    
    return redirect(url_for('contact'))

@app.route('/tech_blog')
def tech_blog():
    # Load data from CSV file
    df = pd.read_csv('data/article_data.csv')
    
    # Convert the dataframe to a list of dictionaries
    data = df.to_dict(orient='records')
    
    # Render the HTML template with all data
    return render_template('Blog/tech.html', data=data)


@app.route('/blog/article')
def article():
    url = request.args.get('url')  # Get the URL parameter from the query string
    if not url:
        return "URL parameter is missing", 400

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the content within the specified div class
        article_content = soup.find('div', class_='article-container')
        if article_content:
            content = str(article_content)
        else:
            return "Content not found", 404

    except requests.RequestException as e:
        return f"An error occurred: {e}", 500

    return render_template('Blog/article.html', content=content)


##########################################################################
@app.route('/subjectexp')
def subjectexp():
    if 'user_id' in session:
        return render_template('subject_expert.html')
    else:
        return redirect(url_for('login'))
    
    
# Load the QA data
with open('qa_data/qa_data.pkl', 'rb') as f:
    qa_dict = pickle.load(f)

# Load CSV data for the question level
df = pd.read_csv('course_data/dsa.csv')  # Update the path to your CSV file

# Initialize Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_similarity(user_answer, correct_answer):
    # Encode sentences
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    correct_embedding = model.encode(correct_answer, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(user_embedding, correct_embedding).item()
    
    # Convert similarity to a score out of 10
    return round(similarity_score * 10, 1)

# Route to home page
@app.route('/dsasubject')
def dsa_subject():
    return render_template('subject_expert/dsa.html')

# Route to get question based on level
@app.route('/get_question', methods=['GET'])
def get_question():
    level = request.args.get('level')
    if not level:
        return jsonify({'error': 'Level parameter is required'})
    
    level_df = df[df['Level'].str.lower() == level.lower()]
    if not level_df.empty:
        question = level_df.iloc[0]['question']
        answers = level_df.iloc[0]['answer']
        index = 0
        length = len(level_df)
        return jsonify({
            'question': question,
            'answers': answers,
            'index': index,
            'length': length
        })
    return jsonify({'error': 'No questions found for this level'})

# Route to check user answer
@app.route('/check_answer', methods=['POST'])
def check_answer():
    data = request.json
    user_answer = data.get('answer')
    correct_answer = data.get('correct_answer')
    
    if not user_answer or not correct_answer:
        return jsonify({'error': 'Answer and correct_answer are required'}), 400
    
    similarity_score = calculate_similarity(user_answer, correct_answer)
    feedback = f"Match Score: {similarity_score}/10"
    correct = similarity_score >= 9.5  # Consider 7 and above as correct
    return jsonify({
        'feedback': feedback,
        'correct_answer': correct_answer,
        'score': similarity_score,
        'correct': correct
    })

# Route to navigate between questions
@app.route('/navigate', methods=['GET'])
def navigate():
    level = request.args.get('level')
    index_str = request.args.get('index')
    
    if not level or not index_str:
        return jsonify({'error': 'Level and index parameters are required'}), 400
    
    try:
        index = int(index_str)
    except ValueError:
        return jsonify({'error': 'Invalid index parameter'}), 400
    
    level_df = df[df['Level'].str.lower() == level.lower()]
    if not level_df.empty:
        if 0 <= index < len(level_df):
            question = level_df.iloc[index]['question']
            answers = level_df.iloc[index]['answer']
            length = len(level_df)
            return jsonify({
                'question': question,
                'answers': answers,
                'index': index,
                'length': length
            })
        return jsonify({'error': 'Index out of range'}), 400
    
    return jsonify({'error': 'No questions found for this level'})


################################################################################################
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

@app.route('/codequestion')
def questions():
    return jsonify(QUESTIONS)



#######################################################################################
csv_file = 'course_data/hr_interview_data.csv'
data = pd.read_csv(csv_file)

# Load the BERT model and tokenizer
model_path = "kaporter/bert-base-uncased-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Initialize the grammar checker
tool = language_tool_python.LanguageTool('en-US')

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('wordnet')

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def calculate_word_similarity(word1, word2):
    syn1 = wn.synsets(word1)
    syn2 = wn.synsets(word2)
    if syn1 and syn2:
        return wn.wup_similarity(syn1[0], syn2[0])
    return 0

def calculate_sentence_similarity(answer1, answer2):
    words1 = word_tokenize(answer1)
    words2 = word_tokenize(answer2)
    score = sum([max([calculate_word_similarity(w1, w2) or 0 for w2 in words2]) for w1 in words1])
    return score / len(words1)

@app.route('/hrai', methods=['GET', 'POST'])
def hrai():
    if request.method == 'POST':
        action = request.form.get('action', 'submit')
        current_index = request.form.get('current_index', 0)
        try:
            current_index = int(current_index)  # Try converting to integer
        except (ValueError, TypeError):
            current_index = 0  # Set a default value if conversion fails


        if action == 'submit':
            user_question = request.form['question']
            user_answer = request.form['answer']

            # Find the closest question in the dataset
            correct_answer = None
            for index, row in data.iterrows():
                if user_question.lower() in row['question'].lower():
                    correct_answer = row['answer']
                    break

            # Grammar check
            grammar_matches = tool.check(user_answer)
            grammar_feedback = [match.message for match in grammar_matches]

            # If we found a matching question, evaluate the answer
            if correct_answer:
                cosine_sim = calculate_cosine_similarity(user_answer, correct_answer)
                word_sim = calculate_sentence_similarity(user_answer, correct_answer)
                score = (cosine_sim + word_sim) / 2 * 10

                feedback = {
                    'score': f"{score:.2f}/10",
                    'correct_answer': correct_answer,
                    'grammar_feedback': grammar_feedback,
                    'cosine_similarity': cosine_sim,
                    'sentence_similarity': word_sim
                }
            else:
                feedback = {
                    'score': "Question not found in the dataset.",
                    'correct_answer': None,
                    'grammar_feedback': [],
                    'cosine_similarity': 0,
                    'sentence_similarity': 0
                }

            return jsonify(feedback)

        elif action == 'next':
            current_index = min(current_index + 1, len(data) - 1)

        elif action == 'previous':
            current_index = max(current_index - 1, 0)

        question = data.iloc[current_index]['question']
        return jsonify({'question': question, 'current_index': current_index})

    # Initial question display
    initial_question = data['question'].iloc[0]
    return render_template('HR_Round/ai_hr.html', question=initial_question, current_index=0)

if __name__ == '__main__':
    app.run(debug=True)
