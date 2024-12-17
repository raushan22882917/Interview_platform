from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify,send_file
import psycopg2
import smtplib
import requests
from groq import Groq
import paypalrestsdk

import json
import re
from datetime import date
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.dates as mdates
from psycopg2.extras import RealDictCursor
import os
import calendar 
import numpy
import matplotlib
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
import json
from email.mime.text import MIMEText
from datetime import datetime,timedelta
import random
from bs4 import BeautifulSoup
import math
from reportlab.lib import colors
import csv
from fpdf import FPDF
from werkzeug.utils import secure_filename
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
from transformers import AutoTokenizer, AutoModel
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import language_tool_python
import json
from psycopg2 import sql, OperationalError
import language_tool_python
import speech_recognition as sr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from fpdf import FPDF
import plotly.graph_objs as go
import plotly.offline as pyo
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
# Directory to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    # Define allowed extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="interview",
        user="postgres",
        password="22882288"
    )
    return conn

# Login Route
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']

#         conn = get_db_connection()
#         cur = conn.cursor()

#         # Fetch user details
#         cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
#         user = cur.fetchone()

#         cur.close()
#         conn.close()

#         if user:
#             session['user_id'] = user[0]
#             session['user_name'] = user[1]
#             session['email'] = user[2]
#             session['profile_image'] = user[4]  # Assuming profile_image is stored in column 4
#             flash('Login successful!', 'success')
#             return redirect(url_for('index'))
#         else:
#             flash('Invalid email or password', 'danger')

#     return render_template('login.html')
# Profile Route
# Database connection credentials
DB_HOST = 'localhost'
DB_NAME = 'interview'
DB_USER = 'postgres'
DB_PASSWORD = '22882288'
DB_PORT = '5432'  # Default PostgreSQL port

def get_db_connection():
    connection = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    return connection


def get_positions():
    """Fetch unique positions from the database."""
    connection = get_db_connection()
    if not connection:
        return []
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT DISTINCT position FROM user_answers;")
        positions = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching positions: {e}")
        positions = []
    finally:
        cursor.close()
        connection.close()
    
    return positions




def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="interview",
            user="postgres",
            password="22882288",
            port="5432"
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_best_similarity_data_by_position(position):
    """Fetch best similarity data for a given position."""
    connection = get_db_connection()
    if not connection:
        return []
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT question_number, best_similarity FROM user_answers WHERE position = %s;", (position,))
        data = cursor.fetchall()
    except Exception as e:
        print(f"Error fetching data for position {position}: {e}")
        data = []
    finally:
        cursor.close()
        connection.close()
    
    return data



def get_user_activity_dates(user_email):
    # Connect to the PostgreSQL database
    connection = psycopg2.connect(
        host="localhost",
        database="interview",  # Replace with your actual database name
        user="postgres",  # Replace with your database username
        password="22882288"  # Replace with your database password
    )

    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # Query to fetch date and average score from the database
    cursor.execute("""
        SELECT DATE(created_at) AS score_date, AVG(score) AS avg_score
        FROM evaluation_results
        WHERE email = %s
        GROUP BY DATE(created_at)
        ORDER BY score_date;
    """, (user_email,))

    # Fetch all results from the query
    evaluation_results = cursor.fetchall()

    # Store the activity data in a dictionary
    activity_data = {}
    for result in evaluation_results:
        score_date = result['score_date']
        avg_score = result['avg_score']
        
        # Categorizing based on average score
        if avg_score >= 90:
            color = 'green'
        elif avg_score >= 70:
            color = 'orange'
        elif avg_score >= 50:
            color = 'yellow'
        else:
            color = 'red'
        
        # Store the score and color for each score_date
        activity_data[score_date] = {"score": round(avg_score, 2), "color": color}

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Return the activity data
    return activity_data


def generate_full_year_calendar(user_email):
    # Get the activity data for the user
    activity_data = get_user_activity_dates(user_email)

    # Define the month names
    month_names = [
        "January", "February", "March", "April", "May", "June", "July", "August", 
        "September", "October", "November", "December"
    ]

    calendar_data = []

    # Loop through all months
    for month_index in range(1, 13):
        month_data = {
            "name": month_names[month_index - 1],
            "days": []
        }

        # Get the number of days in the current month
        num_days_in_month = calendar.monthrange(2024, month_index)[1]

        # Loop through all days in the current month
        for day in range(1, num_days_in_month + 1):
            date_str = f"2024-{month_index:02d}-{day:02d}"

            # Check if this date has activity data
            if date_str in activity_data:
                score = activity_data[date_str]["score"]
                color = activity_data[date_str]["color"]
            else:
                score = ""
                color = 'gray'  # Default for days without data

            # Add day data
            month_data["days"].append({
                "day": day,
                "score": score,
                "color": color
            })

        calendar_data.append(month_data)

    return calendar_data




@app.route('/profile', methods=['GET', 'POST'])
def profile():
    # Check if user is logged in
    if 'user_name' not in session:
        flash('You need to log in first.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'profile_image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['profile_image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['profile_image'] = filename  # Save filename in session
            flash('Profile image updated successfully!', 'success')
            return redirect(url_for('profile'))

    profile_image = session.get('profile_image', None)
    user_email = session.get('email')

    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch educational details for the user
    # Fetch existing educational details
    cursor.execute("SELECT * FROM education WHERE user_email = %s;", (user_email,))
    education_details = cursor.fetchone()

    if request.method == 'POST':
        # Handle form data
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        college_name = request.form.get('college_name')
        dob = request.form.get('dob')
        year_of_study = request.form.get('year_of_study')
        graduation_year = request.form.get('graduation_year')
        linkedin_url = request.form.get('linkedin_url')  # New field
        instagram_url = request.form.get('instagram_url')  # New field

        cursor.execute("""
    INSERT INTO education (user_email, first_name, last_name, college_name, dob, year_of_study, graduation_year, linkedin_url, instagram_url)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (user_email) 
    DO UPDATE SET 
        first_name = EXCLUDED.first_name,
        last_name = EXCLUDED.last_name,
        college_name = EXCLUDED.college_name,
        dob = EXCLUDED.dob,
        year_of_study = EXCLUDED.year_of_study,
        graduation_year = EXCLUDED.graduation_year,
        linkedin_url = EXCLUDED.linkedin_url,
        instagram_url = EXCLUDED.instagram_url,
        updated_at = CURRENT_TIMESTAMP;
""", (user_email, first_name, last_name, college_name, dob, year_of_study, graduation_year, linkedin_url, instagram_url))

        conn.commit()
        flash('Educational details saved successfully!', 'success')

        # Fetch updated educational details
        cursor.execute("SELECT * FROM education WHERE user_email = %s;", (user_email,))
        education_details = cursor.fetchone()

    # Fetch average score and unique positions
    cursor.execute("SELECT AVG(similarity_scores) FROM quiz_reports WHERE email = %s", (user_email,))
    avg_score = cursor.fetchone()[0] or 0

    cursor.execute("SELECT DISTINCT position FROM user_answers WHERE user_email = %s;", (user_email,))
    unique_positions = cursor.fetchall()
    cursor.execute("""
        SELECT position, MAX(total_score) AS max_score, created_at
        FROM user_scores
        GROUP BY position, created_at
        HAVING MAX(total_score) > 45
    """)

    user_scores = cursor.fetchall()
    user_data = []
    all_results = {}
    evaluation_results = []  # Initialize the evaluation results list

    # Extract evaluation results for logged-in user
    cursor.execute("""
        SELECT question_id, title, passed_test_cases_count, created_at, language
        FROM evaluation_results
        WHERE email = %s;
    """, (user_email,))
    evaluation_results = cursor.fetchall()


    graph_html = generate_unique_questions_graph(cursor, user_email)
    total_attempts_count = 0
    scored_above_70_count = 0
    positions_well_prepared = 0
    last_3_scores = []

    for position in unique_positions:
        position_name = position[0]

        # Fetch the best similarity data using a function
        best_similarity_data = get_best_similarity_data_by_position(position_name)

        total_marks = sum([row[1] for row in best_similarity_data])
        total_possible_marks = len(best_similarity_data) * 10
        percentage = round((total_marks / total_possible_marks) * 100, 2) if total_possible_marks > 0 else 0

        # Track attempts and scores for graph
        attempts = []
        summed_scores = []
        current_score = 0
        attempt_counter = 0

        for i in range(len(best_similarity_data)):
            question_number, best_similarity = best_similarity_data[i]
            current_score += best_similarity
            if (i + 1) % 5 == 0:  # Every 5 questions
                attempts.append(f'Attempt {attempt_counter + 1}')
                summed_scores.append(current_score)
                current_score = 0
                attempt_counter += 1

        # Append any remaining score for the last attempt if needed
        if current_score > 0:
            attempts.append(f'Attempt {attempt_counter + 1}')
            summed_scores.append(current_score)

        # Calculate total attempts, top score, and average of last 3 attempts
        total_attempts = len(summed_scores)
        total_attempts_count += total_attempts
        top_score = max(summed_scores) if summed_scores else 0
        last_3_attempts = summed_scores[-3:] if len(summed_scores) >= 3 else summed_scores
        avg_last_3_attempts = round(sum(last_3_attempts) / len(last_3_attempts), 2) if last_3_attempts else 0

        last_3_scores.append(avg_last_3_attempts)

        # Check if the candidate is well prepared
        if avg_last_3_attempts > 35:
            scored_above_70_count += 1

        # Count attempts scoring above 70%
        if any(score > 35 for score in summed_scores):
            scored_above_70_count += 1

        # Overall feedback based on the average of the last 3 attempts
        if avg_last_3_attempts >= 90:
            feedback = "Excellent"
        elif avg_last_3_attempts >= 75:
            feedback = "Very Good"
        elif avg_last_3_attempts >= 50:
            feedback = "Good"
        elif avg_last_3_attempts >= 30:
            feedback = "Need to Practice"
        else:
            feedback = "Need to Practice"

        # Store data for the position
        user_data.append({
            'position': position_name,
            'total_attempts': total_attempts,
            'top_score': top_score,
            'average_last_3_attempts': avg_last_3_attempts,
            'overall_feedback': feedback,
            'attempts': [{'number': attempts[i], 'score': summed_scores[i]} for i in range(total_attempts)],
            'total_marks': total_marks,
            'percentage': percentage
        })

        # Prepare all_results structure
        if position_name not in all_results:
            all_results[position_name] = ([], [])

    positions = fetch_all_positions()

    # Fetch data for each position and prepare results
    for position in positions:
        data = fetch_user_data(position)

        # Divide data into parts
        parts = [data[i:i + 5] for i in range(0, len(data), 5)]
        results = []

        for part_number, part in enumerate(parts, start=1):
            score = sum(row[4] for row in part)  # Sum of best_similarity
            # Fetch created_at from the first row in the part for display
            created_at = part[0][7] if part else "N/A"  # Assuming created_at is in the 8th column
            results.append((position, part_number, score, created_at))

        all_results[position] = (results, parts)


    attended_days = get_user_activity_dates(user_email)

    # Fetch maximum scores for each `question_id` grouped by date
   # Fetch scores for each question_id grouped by date
    cursor.execute("""
        SELECT 
            question_id, 
            DATE(created_at) AS score_date, 
            AVG(score) AS score 
        FROM evaluation_results
        WHERE email = %s
        GROUP BY question_id, score_date  -- Ensure grouping by question_id and date
        ORDER BY score_date, question_id; -- Order by date first, then question_id
    """, (user_email,))
    scores_data = cursor.fetchall()

    # Prepare data for the graph
    scores_by_question = {}
    for question_id, score_date, score in scores_data:
        # Ensure score_date is properly formatted
        scores_by_question.setdefault(question_id, []).append((score_date, score))

    # Create the graph
    graph_html = generate_combined_graph(scores_by_question)

    calendar_data = generate_full_year_calendar(user_email)
    streak_data = get_user_streak_levels(user_email)
    # Calculate the final metrics
    total_positions = len(unique_positions)
    percentage_scored_above_70 = (scored_above_70_count / 50 * 100) if total_attempts_count > 0 else 0
    average_score_last_3 = round(sum(last_3_scores) / len(last_3_scores), 2) if last_3_scores else 0

    return render_template('profile_details.html',
                           profile_image=profile_image,
                           education_details=education_details,
                           average_score=avg_score,
                           user_data=user_data,
                           all_results=all_results,
                           evaluation_results=evaluation_results,  # Add evaluation_results to the context
                           positions_well_prepared=positions_well_prepared,
                           percentage_scored_above_70=percentage_scored_above_70,
                           average_score_last_3=average_score_last_3,
                           attended_days=attended_days,user_scores=user_scores,graph_html=graph_html,streak_data=streak_data,calendar_data=calendar_data) 
    


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_combined_graph(scores_by_question):
    """
    Generate a combined graph with bars for scores by question_id on each date
    and a line chart connecting average scores per date.
    """
    # Prepare data
    dates = sorted({date for scores in scores_by_question.values() for date, _ in scores})  # All unique dates
    question_ids = sorted(scores_by_question.keys())  # All unique question_ids
    
    # Aggregate scores by question_id and date
    scores_by_date = defaultdict(lambda: [0] * len(question_ids))  # Default scores for all question_ids
    for qid, values in scores_by_question.items():
        for date, score in values:
            scores_by_date[date][question_ids.index(qid)] = score  # Place score in the correct position

    # Calculate average score per date for the line graph
    avg_scores = [np.mean(scores) for scores in scores_by_date.values()]
    
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(max(len(dates) * 2, 12), 6))  # Ensure minimum figure width
    
    # Fixed width for each date group and bar width within the group
    max_width_per_date = 200  # Convert 200px to relative figure units
    if len(question_ids) > 0:
        bar_width = max_width_per_date / len(question_ids)  # Divide space for all question_ids
    else:
        bar_width = 0  # or assign a default value depending on your use case

    # X positions for date groups
    x_positions = np.arange(len(dates)) * max_width_per_date  # Ensure fixed spacing between dates

    # Colors for each question_id (using a dynamic color map for more than 10 colors)
    color_palette = plt.cm.get_cmap('tab20', len(question_ids))  # Ensure enough colors
    
    # Plot bars for each question_id on each date
    for i, qid in enumerate(question_ids):
        scores = [scores_by_date[date][i] for date in dates]
        ax.bar(
            x_positions + i * bar_width,  # Shift bars for different question_ids
            scores,
            width=bar_width,
            label=f"Question {qid}",
            color=color_palette(i),
            alpha=0.7,
            edgecolor="black",
        )

    # Add a zigzag line graph for average scores
    ax.plot(
        x_positions + (len(question_ids) - 1) * bar_width / 2,  # Align line at the center of date groups
        avg_scores,
        color="red",
        marker="o",
        label="Average Score",
        linewidth=2,
    )

    # Set x-axis labels for dates
    ax.set_xticks(x_positions + (len(question_ids) - 1) * bar_width / 2)  # Align x-ticks at group centers
    ax.set_xticklabels(dates, rotation=45, ha="right")

    # Labels and title
    ax.set_xlabel("Dates")
    ax.set_ylabel("Scores")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout and show plot
    plt.tight_layout()

    # Convert the graph to a format suitable for HTML display
    from io import BytesIO
    import base64
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    return f"<img src='data:image/png;base64,{graph_url}'/>"

################################################################################################
def get_user_streak_levels(email):
    # Database connection and streak calculation logic (as provided in the earlier code)
    connection = psycopg2.connect(
        host="localhost",
        database="interview",
        user="postgres",
        password="22882288"
    )
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # Fetch user activity data
    cursor.execute("""
        SELECT email, DATE(created_at) AS activity_date, AVG(score) AS avg_score
        FROM evaluation_results
        WHERE email = %s
        GROUP BY email, DATE(created_at)
        ORDER BY activity_date ASC;
    """, (email,))
    user_activity = cursor.fetchall()

    # Initialize streak data with 8 locked levels
    streak_data = [{'level': i + 1, 'status': 'locked'} for i in range(8)]
    streak_level = 0
    consecutive_days = 0
    previous_date = None

    # Process data for streak levels
    for activity in user_activity:
        activity_date = activity['activity_date']
        avg_score = activity['avg_score']

        if avg_score >= 3:
            if previous_date and (activity_date - previous_date).days == 1:
                consecutive_days += 1
            else:
                consecutive_days = 1

            if consecutive_days == 2:
                streak_level += 1
                if streak_level <= 8:
                    streak_data[streak_level - 1]['status'] = 'unlocked'
                consecutive_days = 0

        previous_date = activity_date

    cursor.close()
    connection.close()

    for item in streak_data:
        if item['status'] == 'locked':
            item['icon'] = '🔒'
        elif item['status'] == 'unlocked':
            item['icon'] = '⭐'

    return streak_data
#########################################################################

def get_db_connection():
    # Replace with your own database connection parameters
    conn = psycopg2.connect(
        host='localhost',
        database='interview',
        user='postgres',
        password='22882288'
    )
    return conn

@app.route('/publish', methods=['POST'])
def publishdata():
    if 'user_name' not in session:
        return jsonify({'status': 'error', 'message': 'You need to log in first.'}), 403

    user_email = session.get('email')  # Retrieve email from session
    attempt_number = request.json.get('attempt_number')
    adjusted_percentage = request.json.get('adjusted_percentage')
    status = 'Published'

    # Check for empty or invalid input
    if attempt_number is None or adjusted_percentage is None:
        return jsonify({'status': 'error', 'message': 'Attempt number and adjusted percentage are required.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Insert the published result into the database
        cursor.execute("""
            INSERT INTO published_results (user_email, attempt_number, adjusted_percentage, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_email, attempt_number) 
            DO UPDATE SET adjusted_percentage = EXCLUDED.adjusted_percentage,
                          status = EXCLUDED.status;
        """, (user_email, attempt_number, adjusted_percentage, status))

        conn.commit()
        return jsonify({'status': 'success', 'message': 'Data published successfully!'})
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        cursor.close()
        conn.close()


#################################################################################################################
def generate_unique_questions_graph(cursor, user_email):
    # Fetch data from the database
    cursor.execute("""
        SELECT DATE(created_at) AS created_date, COUNT(DISTINCT question_id) AS unique_questions_count
        FROM evaluation_results
        WHERE email = %s
        GROUP BY created_date
        ORDER BY created_date;
    """, (user_email,))
    
    evaluation_results = cursor.fetchall()

    # Extract the data from the query result
    dates = [row[0] for row in evaluation_results]
    unique_counts = [row[1] for row in evaluation_results]

    # Convert dates to string format (YYYY-MM-DD)
    formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Create the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=formatted_dates, y=unique_counts, mode='lines+markers', name='Unique Questions'))

    # Update layout
    fig.update_layout(title='Count of Unique Questions Over Time',
                      xaxis_title='Date',
                      yaxis_title='Per day question count')

    # Convert the Plotly figure to HTML
    graph_html = fig.to_html(full_html=False)
    
    return graph_html
# Route to handle publishing data


def get_best_similarity_data_by_position(position_name):
    # Replace this function with the actual implementation to fetch best similarity data for the position
    # This is a placeholder example
    return [(i + 1, i * 2) for i in range(10)]  # Sample data: (question_number, best_similarity)
def get_feedback(avg_last_3_attempts):
    if avg_last_3_attempts >= 90:
        return "Excellent"
    elif avg_last_3_attempts >= 75:
        return "Very Good"
    elif avg_last_3_attempts >= 50:
        return "Good"
    elif avg_last_3_attempts >= 30:
        return "Average"
    else:
        return "Need to Practice"
def fetch_all_positions():
    conn = get_db_connection()
    cur = conn.cursor()
    query = "SELECT DISTINCT position FROM user_answers;"
    cur.execute(query)
    positions = cur.fetchall()
    cur.close()
    conn.close()
    return [pos[0] for pos in positions]

# Fetch data for a specific position including created_at
def fetch_user_data(position_name):
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
    SELECT user_email, position, question, user_answer, best_similarity, question_number, attempt_count, created_at
    FROM user_answers
    WHERE position = %s
    ORDER BY best_similarity DESC;  -- Sort by best_similarity
    """
    cur.execute(query, (position_name,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# Generate PDF from the fetched data
def create_pdf(data):
    pdf_filename = "output.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4
    
    # Set background color
    c.setFillColor(colors.yellow)
    c.rect(0, 0, width, height, fill=1)

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2, height - 40, "Sakshatar")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 60, "Welcome to the Sakshatar Experience! 🎉")
    
    # Add a line under the header
    c.setStrokeColor(colors.black)
    c.line(50, height - 70, width - 50, height - 70)

    # Introduction Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 90, "Introduction")
    
    c.setFont("Helvetica", 11)
    intro_text = ("This report contains an overview of the interview questions and the corresponding answers provided. "
                  "The insights gathered aim to enhance the evaluation process and improve overall performance.")
    
    # Split the introduction text to fit within 400 points width
    intro_lines = simpleSplit(intro_text, "Helvetica", 11, 400)
    y_position = height - 110
    for line in intro_lines:
        c.drawString(50, y_position, line)
        y_position -= 14  # Move down for next line

    # Adding questions and answers
    y_position -= 20  # Add space before questions
    for idx, row in enumerate(data):
        question, user_answer = row[2], row[3]
        
        # Question
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.red)
        question_lines = simpleSplit(f"Question {idx + 1}: {question}", "Helvetica-Bold", 12, 400)
        
        for line in question_lines:
            c.drawString(50, y_position, line)
            y_position -= 14  # Move down for next line
        
        # Answer
        c.setFillColor(colors.green)
        c.setFont("Helvetica", 11)
        answer_lines = simpleSplit(f"Answer: {user_answer}", "Helvetica", 11, 400)
        
        for line in answer_lines:
            c.drawString(50, y_position, line)
            y_position -= 14  # Move down for next line

        y_position -= 10  # Add extra space after each question-answer pair

        # Check for page overflow
        if y_position < 50:
            c.showPage()  # Create a new page
            c.setFillColor(colors.yellow)
            c.rect(0, 0, width, height, fill=1)
            y_position = height - 40  # Reset y_position for new page

    # Conclusion Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Conclusion")
    c.setFont("Helvetica", 11)
    conclusion_text = "This report summarizes the insights gained from the interview process."
    conclusion_lines = simpleSplit(conclusion_text, "Helvetica", 11, 400)
    y_position -= 20
    for line in conclusion_lines:
        c.drawString(50, y_position, line)
        y_position -= 14  # Move down for next line

    # Thank You Section
    y_position -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, y_position, "Thank You! 🙏")
    y_position -= 20
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, y_position, "We appreciate your effort and time! 😊")

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.gray)
    footer_text = "This document was generated by Sakshatar."
    c.drawCentredString(width / 2, 30, footer_text)

    c.save()

    return pdf_filename

@app.route('/download/<position_name>/<int:part_number>')
def download_pdf(position_name, part_number):
    data = fetch_user_data(position_name)
    
    # Divide data into parts
    parts = [data[i:i + 5] for i in range(0, len(data), 5)]
    
    # Get the specific part data
    if part_number - 1 < len(parts):
        part_data = parts[part_number - 1]
        pdf_output = create_pdf(part_data)

        return send_file(pdf_output, as_attachment=True, download_name=f"{position_name}_part_{part_number}.pdf", mimetype='application/pdf')

    return "Part not found", 404







# Image Upload Route
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user_email' not in session:
        flash('Please log in to upload an image.', 'warning')
        return redirect(url_for('profile'))

    if 'profile_image' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('profile'))

    file = request.files['profile_image']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('profile'))

    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(f"{session['user_email']}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Update the user's profile image in the database
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET profile_image = %s WHERE email = %s", (filename, session['user_email']))
            conn.commit()

            # Update the session with the new profile image filename
            session['profile_image'] = filename

            flash('Profile image uploaded successfully!', 'success')
        except Exception as e:
            conn.rollback()
            flash('An error occurred while updating your profile image.', 'danger')
            print(f"Error: {e}")
        finally:
            conn.close()

        return redirect(url_for('profile'))

    flash('Invalid file format', 'danger')
    return redirect(url_for('profile'))


# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    # flash('You have been logged out.', 'success')
    return redirect(url_for('index')) 


# Index route
@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Query to count unique users
    cur.execute("SELECT COUNT(DISTINCT email) AS total_users FROM users")
    total_users = cur.fetchone()[0]

    # Query to count total interviews
    # cur.execute("SELECT COUNT(*) AS total_interviews FROM interview")
    # total_interviews = cur.fetchone()[0]

    # Close the cursor and connection
    cur.close()
    conn.close()

    # Render the statistics in the HTML
    return render_template('index.html', total_users=total_users)

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
    
# @app.route('/instructiondetail')
# def int():
#     if 'user_id' in session:
#         return render_template('video_int/inst.html')
#     else:
#         return redirect(url_for('login'))
    
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
    
@app.route('/privacy')
def privacy():
    if 'user_id' in session:
        return render_template('privacy.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/description')
def description():
    if 'user_id' in session:
        return render_template('HR_Round/description.html')
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


##############################################################################
@app.route('/publish_report', methods=['POST'])
def publish_report():
    try:
        data = request.get_json()
        attempt_number = data.get('attempt_number')
        user_email = data.get('user_email')

        # Check if attempt_number is valid
        if not attempt_number or not isinstance(attempt_number, str) or not attempt_number.isdigit():
            return jsonify({'success': False, 'error': 'Invalid attempt number'})

        attempt_number = int(attempt_number)  # Convert to integer

        # Connect to your PostgreSQL database
        conn = psycopg2.connect(
            host="localhost",
            database="interview",
            user="postgres",
            password="22882288"
        )
        cursor = conn.cursor()

        # SQL command to insert the report info
        cursor.execute("""
            INSERT INTO reports (user_email, attempt_number)
            VALUES (%s, %s)
        """, (user_email, attempt_number))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

###############################

@app.route('/questions')
def question_page():
    return render_template('checks/questions.html')


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

################################################### Database connection parameters
DATABASE = {
    'dbname': 'interview',
    'user': 'postgres',
    'password': '22882288',
    'host': 'localhost',
    'port': '5432'
}

def insert_message(name, email, message):
    try:
        conn = psycopg2.connect(**DATABASE)
        cur = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO contact_messages (name, email, message)
            VALUES (%s, %s, %s)
        """)
        cur.execute(insert_query, (name, email, message))
        conn.commit()
        cur.close()
        conn.close()
        print("Message stored successfully!")  # Debugging output
    except OperationalError as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Debugging output
        print(f"Received form data: name={name}, email={email}, message={message}")
        
        insert_message(name, email, message)
        flash('Message sent successfully!')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

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
            # Remove the related articles and breadcrumbs sections
            for related in article_content.find_all('div', class_=['article-related-articles', 'article-breadcrumbs']):
                related.decompose()  # Remove the specified div from the soup
            
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

#######################################################################################



###############################################################################
# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define position names and corresponding CSV files
positions = {
    "Machine Learning": "tech_csv/machine_learning.csv",
    "Backend Developer": "tech_csv/backend_developer.csv",
    "Software Engineer": "tech_csv/questions.csv",
    "Frontend Developer": "tech_csv/frontend_developer_questions.csv",
    "Data Scientist": "tech_csv/data_scientist_questions.csv",
    "DevOps Engineer": "tech_csv/devops_engineer_questions.csv",
    "Mobile App Developer": "tech_csv/mobile_app_developer_questions.csv",
    "Full Stack Developer": "tech_csv/full_stack_developer_questions.csv",
    "Site Reliability Engineer": "tech_csv/site_reliability_engineer_questions.csv",
    "Cloud Architect": "tech_csv/cloud_architect_questions.csv"
}

# Database configuration
DB_HOST = 'localhost'
DB_NAME = 'interview'
DB_USER = 'postgres'
DB_PASSWORD = '22882288'

# Route for the homepage


@app.route('/Tech_round')
def Tech_round():
    if 'user_id' in session:
        return render_template('tech_round/tech_round.html',positions=positions)
    else:
        return redirect(url_for('login'))

# Route for starting a quiz for a specific position
@app.route('/start/<position_name>')
def start_quiz(position_name):
    csv_file = positions.get(position_name)
    if not csv_file:
        return "Position not found", 404
    
    # Load questions from the CSV
    df = pd.read_csv(csv_file)
    questions = df[['Question']].to_dict(orient='records')
    
    # Randomly select 10 questions
    selected_questions = random.sample(questions, 10)

    # Save selected questions and index to session
    session['selected_questions'] = selected_questions
    session['current_index'] = 0
    session['position'] = position_name
    
    return render_template('tech_round/tech_round_quiz.html', position=position_name)

# Route for getting the next question
@app.route('/next_question', methods=['GET'])
def next_question():
    current_index = session.get('current_index', 0)
    selected_questions = session.get('selected_questions', [])
    
    if current_index < len(selected_questions):
        question = selected_questions[current_index]['Question']
        return jsonify({'question': question})
    else:
        return jsonify({'message': 'Quiz complete'}), 200

# Route for submitting an answer
# Assume we have a function to determine the current attempt count for the user
# Function to get the current attempt count based on completed question sets
def get_attempt_count(user_email, position):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        query = """
            SELECT COALESCE(MAX(attempt_count), 0) FROM user_answers
            WHERE user_email = %s AND position = %s
        """
        cursor.execute(query, (user_email, position))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        return result[0] + 1  # Increment for new attempt
    except Exception as e:
        print(f"Error fetching attempt count: {e}")
        return 1  # Default to 1 if there is an error


# Route for submitting an answer
@app.route('/submit_answer', methods=['POST'])
def submit_answer_take():
    data = request.json
    user_answer = data['user_answer']
    position = session.get('position')
    user_email = session.get('user_email')  # Get user email from session
    csv_file = positions.get(position)

    # Ensure CSV file exists
    if not csv_file:
        return jsonify({'error': 'Position not found'}), 404
    
    selected_questions = session.get('selected_questions', [])
    current_index = session.get('current_index', 0)

    if current_index < len(selected_questions):
        question = selected_questions[current_index]['Question']

        # Load correct answers safely
        try:
            df = pd.read_csv(csv_file)
            row = df[df['Question'] == question].iloc[0]
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        web_answer = row['Web Answer']
        gpt35_answer = row['GPT3.5 Answer']
        gpt4_answer = row['GPT4 Answer']
        
        # Calculate similarity with each answer
        user_embedding = model.encode(user_answer)
        web_embedding = model.encode(web_answer)
        gpt35_embedding = model.encode(gpt35_answer)
        gpt4_embedding = model.encode(gpt4_answer)
        
        web_similarity = util.pytorch_cos_sim(user_embedding, web_embedding).item()
        gpt35_similarity = util.pytorch_cos_sim(user_embedding, gpt35_embedding).item()
        gpt4_similarity = util.pytorch_cos_sim(user_embedding, gpt4_embedding).item()
        
        # Determine the best similarity score
        best_similarity = max(web_similarity, gpt35_similarity, gpt4_similarity) * 10

        # Get the current attempt count
        attempt_count = get_attempt_count(user_email, position)

        # Save to database
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO user_answers (user_email, position, question, user_answer, best_similarity, question_number, attempt_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_email, position, question, user_answer, best_similarity, (current_index % 5) + 1, attempt_count))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_error:
            return jsonify({'error': str(db_error)}), 500

        # Update the current index
        session['current_index'] = current_index + 1

        # Check if all questions for this attempt have been answered
        if session['current_index'] % 5 == 0:  # If all questions for the attempt have been answered
            session['current_index'] = 0  # Reset for the next attempt
            
        return jsonify({
            'web_similarity': web_similarity,
            'gpt35_similarity': gpt35_similarity,
            'gpt4_similarity': gpt4_similarity,
            'best_similarity': best_similarity
        })
    else:
        return jsonify({'message': 'Quiz complete'}), 200



@app.route('/progress_table')
def progress_table():
    if 'user_email' not in session:
        flash('You need to log in first.', 'warning')
        return redirect(url_for('login'))

    user_email = session.get('user_email')

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch position, user_answer, question, and best_similarity (score) for the user
    cursor.execute("""
        SELECT position, question, user_answer, best_similarity
        FROM user_answers
        WHERE user_email = %s
        ORDER BY position
    """, (user_email,))
    user_data = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    return render_template('progress_table.html', user_data=user_data)



######################################################################################
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
        {"input": "-1 10\n", "expected_output": "10\n"},
        {"input": "0 100\n", "expected_output": "1\n"},
        {"input": "123 456\n", "expected_output": "56088\n"}
    ]
},

    {
        "title": "Even Number Check",
        "question": "Check if a number is even.",
        "test_cases": [
            {"input": "4\n", "expected_output": "True\n"},
            {"input": "5\n", "expected_output": "True\n"},
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
    if 'email' not in session:  # Redirect if 'email' is not in session
        return redirect(url_for('login'))

    user_email = session['email']
    attempt_status = get_attempt_status(user_email)  # Fetch the attempt status based on the user

    # Render template with attempt_status data
    return render_template('editor/editor.html', attempt_status=attempt_status)



@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'email' not in session:
        return jsonify({"message": "User not logged in."}), 403

    data = request.json
    code = data['code']
    language = data['language']
    class_name = data['class_name']
    question_id = int(data['question_id'])
    user_email = session['email']

    if question_id < 0 or question_id >= len(questions):
        return jsonify({"message": "Invalid question ID."}), 400

    # Check attempt count for the user, question, and current date
    conn = get_db_connection()
    cursor = conn.cursor()
    current_date = date.today()

    cursor.execute(
        """
        SELECT COUNT(*) FROM evaluation_results
        WHERE email = %s AND question_id = %s AND created_at = %s
        """,
        (user_email, str(question_id), current_date)
    )

    attempt_count = cursor.fetchone()[0]

    # If attempt count reaches the limit (2), block further attempts
    if attempt_count >= 2:
        cursor.close()
        conn.close()
        return jsonify({"message": "You have reached the maximum attempt limit for this question today."}), 403

    question = questions[question_id]
    test_cases = question["test_cases"]
    title = question["title"]

    results = []
    all_passed = True  
    passed_count = 0  

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
        compile_command = ['g++', filename, '-o', 'user_code.exe']
        compile_result = subprocess.run(compile_command, capture_output=True, text=True)

        if compile_result.returncode != 0:
            results.append({"output": compile_result.stderr.strip(), "is_correct": False})
            all_passed = False
            save_results_to_db(question_id, title, code, language, user_email, results, all_passed, 0, attempt_count + 1, 0)
            return jsonify({"results": results, "all_passed": all_passed, "message": "Compilation failed."})

        command = ['./user_code.exe']
    elif language == 'java':
        filename = f'{class_name}.java'
        with open(filename, 'w') as f:
            f.write(code)
        compile_command = ['javac', filename]
        compile_result = subprocess.run(compile_command, capture_output=True, text=True)

        if compile_result.returncode != 0:
            results.append({"output": compile_result.stderr.strip(), "is_correct": False})
            all_passed = False
            save_results_to_db(question_id, title, code, language, user_email, results, all_passed, 0, attempt_count + 1, 0)
            return jsonify({"results": results, "all_passed": all_passed, "message": "Compilation failed."})

        command = ['java', '-cp', '.', class_name]

    try:
        for test_case in test_cases:
            input_data = test_case["input"]
            expected_output = test_case["expected_output"]

            result = subprocess.run(command, input=input_data, text=True, capture_output=True, timeout=5)
            output = result.stdout.strip()
            error = result.stderr.strip()

            if error:
                results.append({"output": error, "is_correct": False})
                all_passed = False
            else:
                is_correct = output == expected_output.strip()
                results.append({"output": output, "is_correct": is_correct})
                if is_correct:
                    passed_count += 1
                else:
                    all_passed = False

        # Calculate the score
        score = passed_count * 2

        # Save attempt result and increment attempt count in the database
        save_results_to_db(question_id, title, code, language, user_email, results, all_passed, passed_count, attempt_count + 1, score)

    except subprocess.TimeoutExpired:
        results.append({"output": "Execution timed out.", "is_correct": False})
        all_passed = False
        save_results_to_db(question_id, title, code, language, user_email, results, all_passed, 0, attempt_count + 1, 0)

    finally:
        if os.path.exists(filename):
            os.remove(filename)
        if language == 'cpp' and os.path.exists('user_code.exe'):
            os.remove('user_code.exe')
        if language == 'java' and os.path.exists(f'{class_name}.class'):
            os.remove(f'{class_name}.class')

    return jsonify({
        "message": "Attempt 2 recorded. You have completed your second attempt and Question limit has been reached." if attempt_count == 1 else "Attempt 1 recorded."
    })


# Updated function to save results to the database, including the score
def save_results_to_db(question_id, title, code, language, user_email, results, all_passed, passed_count, attempt_count, score):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO evaluation_results (question_id, title, code, language, email, results, all_passed, passed_test_cases_count, created_at, attempt_count, score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (question_id, title, code, language, user_email, json.dumps(results), all_passed, passed_count, date.today(), attempt_count, score)
        )
        conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        cursor.close()
        conn.close()

        
        

def get_attempt_status(user_email):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get the current date
    current_date = datetime.now().date()

    # Fetch the latest attempt count for each question for the current user on the current date
    cursor.execute(
        """
        SELECT question_id, MAX(attempt_count) AS attempts
        FROM evaluation_results
        WHERE email = %s AND DATE(created_at) = %s
        GROUP BY question_id
        ORDER BY question_id
        LIMIT 5
        """,
        (user_email, current_date)
    )

    attempts_data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Prepare attempt data for rendering in HTML
    attempt_status = {i: 0 for i in range(5)}  # Initialize with 0 attempts for questions 0-4
    for question_id, attempts in attempts_data:
        attempt_status[question_id] = attempts

    return attempt_status

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()

    email = session['email']
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('search', '', type=str)

    entries_per_page = 50
    offset = (page - 1) * entries_per_page
    current_date = datetime.now().strftime('%Y-%m-%d')

    try:
        # Fetch leaderboard data for each question with the highest score per question for the current day
        cursor.execute(
            """
            SELECT email, question_id, title, MAX(score) AS best_score, passed_test_cases_count, created_at, language
            FROM evaluation_results
            WHERE email ILIKE %s
            AND DATE(created_at) = %s
            GROUP BY email, question_id, title, passed_test_cases_count, created_at, language
            ORDER BY best_score DESC
            LIMIT %s OFFSET %s
            """,
            ('%' + search_query + '%', current_date, entries_per_page, offset)
        )
        results = cursor.fetchall()

        # Fetch the total count of distinct question_id entries for pagination
        cursor.execute(
            """
            SELECT COUNT(DISTINCT question_id)
            FROM evaluation_results
            WHERE email ILIKE %s
            AND DATE(created_at) = %s
            """,
            ('%' + search_query + '%', current_date)
        )
        total_entries = cursor.fetchone()[0]
        total_pages = (total_entries + entries_per_page - 1) // entries_per_page

        # Fetch the highest score for each question_id attempted by the user for that day and sum it up
        cursor.execute(
            """
            SELECT SUM(max_score) AS total_score
            FROM (
                SELECT MAX(score) AS max_score
                FROM evaluation_results
                WHERE email = %s
                AND DATE(created_at) = %s
                GROUP BY question_id
            ) AS daily_best_scores
            """,
            (email, current_date)
        )
        total_score = cursor.fetchone()[0] or 0

        # Format leaderboard data
        leaderboard_data = [
            {
                "email": row[0],
                "question_id": row[1],
                "title": row[2],
                "best_score": row[3],
                "passed_test_cases_count": row[4],
                "created_at": row[5].strftime('%Y-%m-%d') if isinstance(row[5], datetime) else row[5],
                "language": row[6]
            } for row in results
        ]
        
        

    except Exception as e:
        print(f"Error fetching leaderboard: {e}")
        leaderboard_data = []
        total_score = 0
        total_pages = 0
    finally:
        cursor.close()
        conn.close()

    return render_template(
        'editor/leaderboard.html',
        leaderboard=leaderboard_data,
        page=page,
        total_pages=total_pages,
        search=search_query,
        total_score=total_score
    )



# Updated function to save results to the database
# def save_results_to_db(question_id, title, code, language, user_email, results, all_passed, passed_count):
#     conn = get_db_connection()
#     cursor = conn.cursor()
    
#     try:
#         cursor.execute(
#             """
#             INSERT INTO evaluation_results (question_id, title, code, language, email, results, all_passed, passed_test_cases_count, created_at)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#             """,
#             (question_id, title, code, language, user_email, json.dumps(results), all_passed, passed_count, date.today())
#         )
#         conn.commit()
#     except Exception as e:
#         print(f"Error saving to database: {e}")
#     finally:
#         cursor.close()
#         conn.close()

# Updated login route to ensure consistency in session keys
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch user details
        cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cur.fetchone()

        cur.close()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            session['email'] = user[2]  # Ensure consistency here
            session['profile_image'] = user[4]
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

        
#############################################################################

@app.route('/interview')
def interview():
    if 'user_id' in session:
        user_id = session['user_id']
        
        # Retrieve user email from the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        email = user[0] if user else None
        
        # Check if the user is already registered
        cur.execute("SELECT * FROM interview_registrations WHERE user_id = %s", (user_id,))
        registration = cur.fetchone()
        cur.close()
        conn.close()

        is_registered = registration is not None
        
        # Get the current time
        current_time = datetime.now()
        return render_template('editor/interview.html', is_registered=is_registered, email=email, current_time=current_time)
    else:
        return redirect(url_for('login'))

# Route to handle interview registration
@app.route('/register_interview', methods=['POST'])
def register_interview():
    if 'user_id' in session:
        user_id = session['user_id']
        
        # Retrieve user email from the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        email = user[0] if user else None

        if not email:
            return jsonify({"error": "User email not found"}), 400
        
        current_time = datetime.now()

        # Check if the registration is allowed based on the current time
        if current_time.hour >= 20:
            # Allow user to register for the next day
            register_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save the registration details to the database
            cur.execute("INSERT INTO interview_registrations (user_id, email, register_time) VALUES (%s, %s, %s)",
                        (user_id, email, register_time))
            conn.commit()
            cur.close()
            conn.close()

            return redirect(url_for('interview'))
        elif not (current_time.hour >= 20 and registration):
            return jsonify({"error": "Registration is closed until 8 PM."}), 400
        
        cur.close()
        conn.close()

        return redirect(url_for('interview'))
    else:
        return redirect(url_for('login'))
    
    
###############################################################################################
# Database connection setup
# Function to get the database connection
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="interview",
        user="postgres",
        password="22882288"
    )
    return conn

# User page route
# @app.route('/user')
# def user():
#     if 'user_email' not in session:
#         return redirect(url_for('login'))  # Redirect to login if not logged in

#     conn = get_db_connection()
#     cur = conn.cursor()

#     # Fetch all notifications for the logged-in user using the session email
#     cur.execute("""
#         SELECT n.id, n.content, n.subject, n.timestamp, un.is_read 
#         FROM notifications n
#         JOIN user_notifications un ON n.id = un.notification_id
#         WHERE un.user_email = %s
#         ORDER BY n.timestamp DESC
#     """, (session['user_email'],))
#     notifications = cur.fetchall()

#     # Fetch the number of unread notifications for showing the dot
#     cur.execute("""
#         SELECT COUNT(*) 
#         FROM user_notifications 
#         WHERE is_read = FALSE AND user_email = %s
#     """, (session['user_email'],))
#     unread_count = cur.fetchone()[0]

#     cur.close()
#     conn.close()

#     # Convert timestamps to human-readable format
#     notifications_data = [
#         (
#             notification[0],  # Notification ID
#             notification[1],  # Notification content
#             notification[2],  # Notification subject
#             notification[3].strftime('%Y-%m-%d %H:%M:%S'),  # Timestamp
#             notification[4],  # Is read status
#             time_ago(notification[3])  # Time ago
#         )
#         for notification in notifications
#     ]

#     return render_template('notification/user.html', notifications=notifications_data, unread_count=unread_count)



# Create a context processor to make unread_count available globally
@app.context_processor
def inject_unread_count():
    unread_count = 0  # Default to 0 if no user is logged in
    if 'user_email' in session:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch the number of unread notifications for the logged-in user
        cur.execute("""
            SELECT COUNT(*) 
            FROM user_notifications 
            WHERE is_read = FALSE AND user_email = %s
        """, (session['user_email'],))
        unread_count = cur.fetchone()[0]

        cur.close()
        conn.close()

    return {'unread_count': unread_count}

@app.route('/user')
def user():
    if 'user_email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all notifications for the logged-in user using the session email
    cur.execute("""
        SELECT n.id, n.content, n.subject, n.timestamp, un.is_read 
        FROM notifications n
        JOIN user_notifications un ON n.id = un.notification_id
        WHERE un.user_email = %s
        ORDER BY n.timestamp DESC
    """, (session['user_email'],))
    notifications = cur.fetchall()

    cur.close()
    conn.close()

    # Convert timestamps to human-readable format
    notifications_data = [
        (
            notification[0],  # Notification ID
            notification[1],  # Notification content
            notification[2],  # Notification subject
            notification[3].strftime('%Y-%m-%d %H:%M:%S'),  # Timestamp
            notification[4],  # Is read status
            time_ago(notification[3])  # Time ago
        )
        for notification in notifications
    ]

    return render_template('notification/user.html', notifications=notifications_data)
def time_ago(timestamp):
    now = datetime.now()
    delta = now - timestamp
    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.seconds // 3600 > 0:
        return f"{delta.seconds // 3600} hour{'s' if delta.seconds // 3600 > 1 else ''} ago"
    elif delta.seconds // 60 > 0:
        return f"{delta.seconds // 60} minute{'s' if delta.seconds // 60 > 1 else ''} ago"
    else:
        return "Just now"

# Mark notification as read
@app.route('/mark_as_read/<int:notification_id>')
def mark_as_read(notification_id):
    if 'user_email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    conn = get_db_connection()
    cur = conn.cursor()

    # Update user_notifications table to mark the notification as read
    cur.execute("""
        UPDATE user_notifications 
        SET is_read = TRUE 
        WHERE notification_id = %s AND user_email = %s
    """, (notification_id, session['user_email']))
    
    conn.commit()
    cur.close()
    conn.close()
    
    return "", 204  # Return a no-content response

# Admin login route
@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        email = request.form['email']
        # Check if email matches admin credentials
        if email in ['raushan22882917@gmail.com', 'raushan2288.jnvbanka@gmail.com']:
            session['email'] = email
            return redirect(url_for('admin'))  # Redirect to admin dashboard
        else:
            flash('Unauthorized access!')
            return redirect(url_for('login_admin'))  # Stay on login if unauthorized

    return render_template('notification/login.html')

# Admin page route
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'email' not in session or session['email'] not in ['raushan22882917@gmail.com', 'raushan2288.jnvbanka@gmail.com']:
        return redirect(url_for('login_admin'))  # Redirect if not an admin
    
    # Handle notification form submission
    if request.method == 'POST':
        # Admin is sending a notification
        if 'subject' in request.form and 'notification' in request.form:
            notification_subject = request.form['subject']
            notification_content = request.form['notification']
            conn = get_db_connection()
            cur = conn.cursor()

            # Insert the notification into the notifications table
            cur.execute("INSERT INTO notifications (subject, content, timestamp) VALUES (%s, %s, %s) RETURNING id", 
                        (notification_subject, notification_content, datetime.now()))
            notification_id = cur.fetchone()[0]

            # Assign notification to all users
            cur.execute("INSERT INTO user_notifications (notification_id, user_email, is_read) SELECT %s, email, FALSE FROM users", 
                        (notification_id,))
            conn.commit()
            cur.close()
            conn.close()
            flash('Notification sent successfully!')

        # Handle payment form submission
        elif 'email' in request.form and 'amount' in request.form:
            email = request.form['email']
            amount = request.form['amount']
            date = request.form['date']
            months = request.form['months']
            action = request.form['action']

            # Insert payment data into the database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO payments (email, amount, date, months)
                VALUES (%s, %s, %s, %s)
            """, (email, amount, date, months))
            conn.commit()
            cur.close()
            conn.close()

            # Redirect based on the action (Save or Save and Next)
            if action == 'save':
                return redirect(url_for('admin'))
            elif action == 'save_next':
                return redirect(url_for('admin'))

    # Fetch the saved payments to display in the table
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT email, amount, date, months FROM payments")
    payments = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('notification/admin.html', payments=payments)

# Admin logout route
@app.route('/logout_admin')
def logout_admin():
    session.pop('email', None)  # Clear the session data for the admin
    flash('You have been logged out.')
    return redirect(url_for('login_admin'))  # Redirect to admin login page


######################################################################################
# Database connection (reuse this function if it's in your code)
def connect_db():
    conn = psycopg2.connect(
        host="localhost",
        database="interview",
        user="postgres",
        password="22882288"
    )
    return conn

# New route to check if user's email exists in user_answers
@app.route('/check_email_in_answers', methods=['POST'])
def check_email_in_answers():
    # Check if user is logged in
    if 'user_email' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user_email = session['user_email']  # Get user email from session
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Check if the email exists in user_answers
        cur.execute("SELECT COUNT(*) FROM user_answers WHERE email = %s", (user_email,))
        row_count = cur.fetchone()[0]

        if row_count > 0:
            # Email exists in user_answers
            return jsonify({'status': 'error', 'message': 'Your free trial has ended. Please proceed to payment.'})
        else:
            # Allow user to proceed
            return jsonify({'status': 'success', 'message': 'Proceed to the next page.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Database error: {e}'})
    finally:
        cur.close()
        conn.close()
        
        
        
################################################################################################
db_config = {
    'host': 'localhost',
    'database': 'interview',
    'user': 'postgres',
    'password': '22882288'
}

def get_db_connection():
    return psycopg2.connect(**db_config)
@app.route('/forum', methods=['GET', 'POST'])
def forum():
    if not session.get('user_name'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        content = request.form['content']
        user_id = session['user_id']
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO questions (user_id, content) VALUES (%s, %s)",
                    (user_id, content)
                )
                conn.commit()

    # Retrieve search term if it exists
    search_term = request.args.get('search', '').lower()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT q.id, q.content, q.post_date, u.email 
                FROM questions q 
                JOIN users u ON q.user_id = u.id
                ORDER BY q.post_date DESC
            """)
            questions = cursor.fetchall()

            question_responses = {}
            for question in questions:
                cursor.execute("""
                    SELECT r.id, r.content, r.post_date, u.email, 
                           COUNT(CASE WHEN re.type = 'like' THEN 1 END) AS likes,
                           COUNT(CASE WHEN re.type = 'dislike' THEN 1 END) AS dislikes
                    FROM responses r
                    JOIN users u ON r.user_id = u.id
                    LEFT JOIN reactions re ON r.id = re.response_id
                    WHERE r.question_id = %s
                    GROUP BY r.id, u.email
                    ORDER BY r.post_date ASC
                """, (question[0],))
                question_responses[question[0]] = cursor.fetchall()

    # Filter questions based on search term
    if search_term:
        filtered_questions = [
            q for q in questions if search_term in q[1].lower()  # Filter by question content
        ]
    else:
        filtered_questions = questions

    # Return filtered questions to the template
    return render_template('forum/forum.html', questions=filtered_questions, question_responses=question_responses)


@app.route('/response/<int:question_id>', methods=['POST'])
def response(question_id):
    content = request.form['content']
    user_id = session['user_id']

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO responses (question_id, user_id, content) VALUES (%s, %s, %s)",
                (question_id, user_id, content)
            )
            conn.commit()
    
    return redirect(url_for('forum'))

@app.route('/like_dislike/<int:response_id>/<action>', methods=['POST'])
def like_dislike(response_id, action):
    user_id = session['user_id']
    
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM reactions WHERE response_id = %s AND user_id = %s",
                (response_id, user_id)
            )
            if action in ['like', 'dislike']:
                cursor.execute(
                    "INSERT INTO reactions (response_id, user_id, type) VALUES (%s, %s, %s)",
                    (response_id, user_id, action)
                )
            conn.commit()
    return redirect(url_for('forum'))


#########################################################################
# PostgreSQL connection details
# Configure image upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DATABASE_CONFIG = {
    'host': 'localhost',
    'database': 'interview',
    'user': 'postgres',
    'password': '22882288'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    return psycopg2.connect(
        host=DATABASE_CONFIG['host'],
        database=DATABASE_CONFIG['database'],
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password']
    )

@app.route('/discussions')
def discussions():
    if 'user_id' in session:
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch discussions with user profile image and uploaded image if available
        cur.execute('''
            SELECT discussions.email, discussions.message, discussions.timestamp, users.profile_image, discussions.image_filename
            FROM discussions
            JOIN users ON discussions.email = users.email
            ORDER BY discussions.timestamp ASC
        ''')
        discussions = cur.fetchall()

        cur.close()
        conn.close()
        return render_template('editor/disscussions.html', discussions=discussions)
    
    return redirect(url_for('login'))

@app.route('/post_discussion', methods=['POST'])
def post_discussion():
    if 'user_id' in session:
        message = request.form['message']
        email = session['email']
        image_filename = None

        # Handle image upload
        if 'image' in request.files:
            image = request.files['image']
            if image and allowed_file(image.filename):
                image_filename = secure_filename(image.filename)
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO discussions (email, message, image_filename) VALUES (%s, %s, %s)",
            (email, message, image_filename)
        )
        conn.commit()
        cur.close()
        conn.close()

        flash('Discussion posted successfully!', 'success')
        return redirect(url_for('discussions'))

@app.route('/delete_message', methods=['POST'])
def delete_message():
    if 'user_id' in session:
        data = request.get_json()
        timestamp = data.get('timestamp')

        conn = get_db_connection()
        cur = conn.cursor()

        # Only delete if the message belongs to the user
        cur.execute(
            "DELETE FROM discussions WHERE timestamp = %s AND email = %s RETURNING *",
            (timestamp, session['email'])
        )
        deleted_message = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return jsonify(success=bool(deleted_message))
    return jsonify(success=False), 403

@app.route('/post_reply')
def post_reply():
    if 'user_id' in session:
        reply_message = request.args.get('message')
        email = session['email']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO discussions (email, message) VALUES (%s, %s)", (email, reply_message))
        conn.commit()
        cur.close()
        conn.close()

        flash('Reply posted successfully!', 'success')
        return redirect(url_for('discussions'))
    return redirect(url_for('login'))

######################################################################################

    
########################################################################################
# Database connection function
# Database connection function
# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        dbname="interview", 
        user="postgres", 
        password="22882288", 
        host="localhost", 
        port="5432"
    )
    return conn

@app.route('/peerQUESTION')
def peerQUESTION():
    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetching problems
    cur.execute("SELECT * FROM problems LIMIT 5;")
    problems = cur.fetchall()

    # Fetching all requests
    cur.execute("""
        SELECT pr.id, pr.problem_id, u1.username AS user_name, u2.username AS partner_name, pr.status
        FROM problem_requests pr
        JOIN users u1 ON pr.user_id = u1.id
        JOIN users u2 ON pr.partner_id = u2.id;
    """)
    requests = cur.fetchall()  # Fetching all requests

    cur.close()
    conn.close()

    return render_template('peer/home.html', problems=problems, requests=requests)


@app.route('/send_request/<int:problem_id>', methods=['POST'])
def send_request(problem_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    conn = get_db_connection()
    cur = conn.cursor()

    # Get user and their potential partner
    cur.execute("SELECT * FROM users WHERE id != %s AND id NOT IN (SELECT partner_id FROM problem_requests WHERE problem_id = %s);", 
                (session['user_id'], problem_id))
    potential_partners = cur.fetchall()

    if potential_partners:
        partner = potential_partners[0]  # Select the first available partner
        # Create a new request entry in the database
        cur.execute("""
            INSERT INTO problem_requests (problem_id, user_id, partner_id, status) 
            VALUES (%s, %s, %s, 'pending');
        """, (problem_id, session['user_id'], partner[0]))
        conn.commit()

    cur.close()
    conn.close()

    return jsonify({'message': 'Request Sent', 'status': 'success'})


@app.route('/accept_request/<int:request_id>', methods=['POST'])
def accept_request(request_id):
    if 'user_id' not in session:
        flash("You need to log in to accept requests.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    cur = conn.cursor()

    # Update request status to 'ready'
    cur.execute(
        "UPDATE problem_requests SET status = 'ready' WHERE id = %s AND (user_id = %s OR partner_id = %s);",
        (request_id, session['user_id'], session['user_id'])
    )
    conn.commit()

    cur.close()
    conn.close()

    flash("Request accepted successfully! You can now start the session.", "success")
    return redirect(url_for('peerQUESTION'))

@app.route('/peereditor/<int:problem_id>/<user_name>/<partner_name>', methods=['GET'])
def peereditor(problem_id, user_name, partner_name):
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    conn = get_db_connection()
    cur = conn.cursor()

    # Get the problem details
    cur.execute("SELECT * FROM problems WHERE id = %s;", (problem_id,))
    problem = cur.fetchone()

    cur.close()
    conn.close()

    return render_template('peer/editor.html', problem=problem, user_name=user_name, partner_name=partner_name)


@app.route('/save_code', methods=['POST'])
def save_code():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    problem_id = request.form.get('problem_id')  # Problem ID
    user_code = request.form.get('code')  # User code
    progress = int(request.form.get('progress'))  # User progress

    conn = get_db_connection()
    cur = conn.cursor()

    # Check if submission exists
    cur.execute("""
        SELECT id FROM submissions WHERE problem_id = %s AND user_id = %s;
    """, (problem_id, session['user_id']))
    existing_submission = cur.fetchone()

    if existing_submission:
        # Update existing submission
        cur.execute("""
            UPDATE submissions
            SET code = %s, progress = %s, updated_at = current_timestamp
            WHERE id = %s;
        """, (user_code, progress, existing_submission[0]))
    else:
        # Insert new submission
        cur.execute("""
            INSERT INTO submissions (problem_id, user_id, code, progress)
            VALUES (%s, %s, %s, %s);
        """, (problem_id, session['user_id'], user_code, progress))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'message': 'Code saved successfully', 'status': 'success'})


@app.route('/get_output/<int:request_id>', methods=['GET'])
def get_output(request_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch the output for both users
    cur.execute("""
        SELECT u1.id AS user_id, u1.username, s.code, s.output 
        FROM submissions s
        JOIN users u1 ON s.user_id = u1.id
        WHERE s.problem_id = (SELECT problem_id FROM problem_requests WHERE id = %s) 
        AND u1.id IN (SELECT user_id FROM problem_requests WHERE id = %s) 
        """, (request_id, request_id))

    outputs = cur.fetchall()

    output_data = {
        'user_output': outputs[0] if outputs else None,
        'partner_output': outputs[1] if len(outputs) > 1 else None,
    }

    cur.close()
    conn.close()

    return jsonify({'output_data': output_data})


################################################################
def get_db_connection():
    return psycopg2.connect(
        dbname="interview",
        user="postgres",
        password="22882288",
        host="localhost",
        port="5432"
    )

@app.route('/hr_round', methods=['GET', 'POST'])
def hr_round():
    if request.method == 'POST':
        # Get the form data
        question_text = request.form['question_text']
        question_type = request.form['question_type']
        difficulty_level = request.form['difficulty_level']
        category = request.form['category']
        subtopic = request.form['subtopic']
        job_role = request.form['job_role']
        experience_level = request.form['experience_level']
        
        # Assuming the email of the logged-in user is stored in the session
        user_email = session.get('email')
        
        # Check if user_email exists
        if not user_email:
            flash("User is not logged in.", "danger")
            return redirect(url_for('login'))  # Redirect to login page if email is not found
        
        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert the question into the database, including the user_email
        try:
            cur.execute('''INSERT INTO intr_questions (question_text, question_type, difficulty_level, category, 
                           subtopic, job_role, experience_level, user_email, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                        (question_text, question_type, difficulty_level, category, subtopic, job_role, 
                         experience_level, user_email, datetime.now()))
            conn.commit()
            flash("Question saved successfully!", "success")
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
        finally:
            cur.close()
            conn.close()
        
        return redirect(url_for('hr_round'))  # Redirect after form submission
    
    # Fetch the questions from the database along with the publisher's email
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''SELECT question_text, question_type, difficulty_level, category, subtopic, 
                          job_role, experience_level, user_email, created_at FROM intr_questions''')
    questions = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('publish/form.html', email=session.get('email'), questions=questions)
#############################################################################
# Initialize Groq client
client = Groq(api_key="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")

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",      # Database server
        database="interview",  # Database name
        user="postgres",       # Your database username
        password="22882288"    # Your database password
    )

# Function to format questions based on company name
def format_questions(company_name):
    return [
        f"Can you tell me about yourself?",
        f"Why do you want to work at {company_name}?",
        f"What are your greatest strengths?",
        f"Where do you see yourself in five years?",
        f"Why should we hire you over other candidates?",
        f"Describe a challenging situation you faced and how you resolved it.",
        f"What motivates you to perform well at work?",
        f"How do you handle criticism or feedback from your manager?",
        f"Can you share an example of how you worked effectively in a team?",
        f"What is your approach to time management and meeting deadlines?"
    ]

# Scoring based on categories
CATEGORY_SCORES = {
    "Bad": 1,
    "Need Improvement": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5,
    "Awesome": 5
}

# Function to evaluate user answer using Groq API
def evaluate_answer_hr(user_answer, question_text):
    # Sending the question and user answer to Groq model for evaluation
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that evaluates answers to interview questions. "
                           "Provide feedback on the answer based on the following categories: "
                           "Good, Bad, Very Good, Excellent, Awesome, and Need Improvement. "
                           "Also, assign a score between 1 to 5 based on the answer quality."
            },
            {
                "role": "user",
                "content": f"Question: {question_text}\nAnswer: {user_answer}"
            }
        ],
        model="llama3-8b-8192",
    )

    # Extracting the response from the API
    feedback = chat_completion.choices[0].message.content
    return feedback

# Function to store data in the database
def store_in_db(company_name, question, user_answer, feedback, score, user_email):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    cursor.execute(
        """
        INSERT INTO interview_data (company_name, question, user_answer, feedback, score, user_email)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (company_name, question, user_answer, feedback, score, user_email)
    )
    
    connection.commit()
    cursor.close()
    connection.close()

@app.route('/hr_index')
def hr_index():
    # Check if 'email' exists in the session
    if 'email' not in session:
        return jsonify({
            'message': 'You need to log in first.'
        }), 401  # Unauthorized

    # Fetch the email from the session
    user_email = session['email']
    
    # Pass the email to the template for rendering
    return render_template('HR_Round/index.html', user_email=user_email)



def check_subscription(user_email):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Check if user has more than 2 entries in interview_data
    cursor.execute("SELECT COUNT(*) FROM interview_data WHERE user_email = %s", (user_email,))
    count = cursor.fetchone()[0]

    if count > 2:
        # Check for active subscription
        cursor.execute("SELECT * FROM payments WHERE email = %s ORDER BY date DESC LIMIT 1", (user_email,))
        subscription = cursor.fetchone()
        
        if subscription:
            # Convert subscription_date to datetime if it is not already
            subscription_date = subscription['date']
            if isinstance(subscription_date, datetime):
                # Already datetime
                pass
            else:
                # Convert date to datetime (assuming it is a datetime.date object)
                subscription_date = datetime.combine(subscription_date, datetime.min.time())
                
            # Calculate the expiration date
            months = subscription['months']
            expiration_date = subscription_date + timedelta(days=months * 30)
            today = datetime.today()

            if expiration_date >= today:
                remaining_days = (expiration_date - today).days
                return 'active', remaining_days
            else:
                return 'expired', None
        else:
            return 'no_subscription', None
    else:
        return 'no_data', None


@app.route('/submit_answer_hr', methods=['POST'])
def submit_answer():
    if 'email' not in session:
        return jsonify({
            'message': 'You need to log in first.'
        }), 401  # Unauthorized

    user_email = session['email']  # Retrieve email from session
    company_name = request.form['company_name']
    question_index = int(request.form['question_index'])
    user_answer = request.form['user_answer']

    # Check subscription status
    subscription_status, remaining_days = check_subscription(user_email)

    if subscription_status == 'no_subscription':
        return jsonify({
            'message': 'You have not taken a subscription. Please subscribe to continue.'
        }), 403  # Forbidden

    elif subscription_status == 'expired':
        return jsonify({
            'message': 'Your subscription has expired. Please renew your subscription.'
        }), 403  # Forbidden

    # Proceed with storing the interview data if subscription is active
    questions = format_questions(company_name)
    question_text = questions[question_index]
    
    # Evaluate answer
    feedback = evaluate_answer_hr(user_answer, question_text)
    
    # Score based on feedback (simplified)
    score = 2  # Default score (you can improve this logic)
    if "Bad" in feedback:
        score = 1
    elif "Need Improvement" in feedback:
        score = 2
    elif "Good" in feedback:
        score = 3
    elif "Very Good" in feedback:
        score = 4
    elif "Excellent" in feedback or "Awesome" in feedback:
        score = 5
    
    # Store in DB with user email
    store_in_db(company_name, question_text, user_answer, feedback, score, user_email)
    
    # Return a success response
    return jsonify({
        'message': 'Your answer has been submitted successfully.'
    })


@app.route('/HrHome')
def Hr_home():
    if 'email' not in session:
        return redirect('/login')

    user_email = session['email']
    
    # Check subscription status
    subscription_status, remaining_days = check_subscription(user_email)

    if subscription_status == 'active':
        return render_template('HR_Round/index.html', button_message="Start", remaining_days=remaining_days)
    elif subscription_status == 'expired':
        return render_template('subscribe.html', button_message="Renew Subscription")
    else:
        return render_template('HR_Round/subscribe.html', button_message="Subscribe Now")
##################################################################################
# PayPal configuration
paypalrestsdk.configure({
    "mode": "sandbox",  # Use 'live' for production
    "client_id": "AVl3kW1ZH4Tq8HG34uKdfAmQHmoLnx7eYiYrELvytySETUfm9r4vFdzNd43T8vUbhIgAhUYPCcpXl4UC",
    "client_secret": "EPLaNfxteaTDI8TE2cvG_GdTjoUi3uZW0_03Fl7-uX1ggAcUGZGeigEXv7iX4YTc-QdWwla979VcUcmu"
})

# Database connection
# Database connection function
def get_db_connection():
    return psycopg2.connect(
        host="localhost",      # Database server
        database="interview",  # Database name
        user="postgres",       # Your database username
        password="22882288"    # Your database password
    )

# Route to render subscription page
@app.route('/subscribe')
def subscribe():
    if 'user_name' not in session:
        return render_template('subscribe.html')  # Update this with your actual HTML file
    else:
        return render_template('index.html')  # Ensure .html extension is included


# Route to handle PayPal payment creation
@app.route('/payment', methods=['POST'])
def payment():

    payment = paypalrestsdk.Payment({
        "intent": "sale",
        "payer": {
            "payment_method": "paypal"},
        "redirect_urls": {
            "return_url": "http://localhost:3000/payment/execute",
            "cancel_url": "http://localhost:3000/"},
        "transactions": [{
            "item_list": {
                "items": [{
                    "name": "testitem",
                    "sku": "12345",
                    "price": "500.00",
                    "currency": "USD",
                    "quantity": 1}]},
            "amount": {
                "total": "500.00",
                "currency": "USD"},
            "description": "This is the payment transaction description."}]})

    if payment.create():
        print('Payment success!')
    else:
        print(payment.error)

    return jsonify({'paymentID' : payment.id})

@app.route('/execute', methods=['POST'])
def execute():
    success = False

    payment = paypalrestsdk.Payment.find(request.form['paymentID'])

    if payment.execute({'payer_id' : request.form['payerID']}):
        print('Execute success!')
        success = True
    else:
        print(payment.error)

    return jsonify({'success' : success})

# Route to handle UID submission (optional for tracking)
@app.route('/submit_uid', methods=['POST'])
def submit_uid():
    # Get email from session (ensure user is logged in)
    email = session.get('email')
    if not email:
        return jsonify({'error': 'User not logged in'}), 400

    uid = request.form['uid']

    if not uid:
        return jsonify({'error': 'UID is required'}), 400

    # Store the UID and email in the database for tracking
    connection = get_db_connection()
    cursor = connection.cursor()

    # Insert into the user_subscriptions table
    cursor.execute("""
        INSERT INTO user_subscriptions (email, uid)
        VALUES (%s, %s)
    """, (email, uid))

    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'message': 'UID submitted successfully'}), 200


#####################################################################################



if __name__ == '__main__':
    app.run(debug=True)

