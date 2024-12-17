from flask import Flask, render_template, send_file
import psycopg2
from fpdf import FPDF
import io
import os

app = Flask(__name__)

# Database connection details (replace with your actual credentials)
DB_HOST = "localhost"
DB_NAME = "interview"
DB_USER = "postgres"
DB_PASS = "22882288"

# Connect to PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

# Fetch all unique positions from the database
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
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    # Add each question and answer in the specified format
    for row in data:
        question, user_answer = row[2], row[3]
        pdf.multi_cell(0, 10, f"Question: {question}\nAnswer: {user_answer}\n\n")  # Multi-cell for line breaks

    # Save PDF to a temporary file
    pdf_filename = "temp.pdf"
    pdf.output(pdf_filename)

    # Read the PDF file into a BytesIO object
    pdf_output = io.BytesIO()
    with open(pdf_filename, 'rb') as f:
        pdf_output.write(f.read())
    
    # Clean up the temporary file
    os.remove(pdf_filename)

    pdf_output.seek(0)
    return pdf_output

@app.route('/')
def index():
    positions = fetch_all_positions()
    all_results = {}

    # Fetch data for each position and prepare results
    for position in positions:
        data = fetch_user_data(position)

        # Divide data into parts
        parts = [data[i:i + 5] for i in range(0, len(data), 5)]
        results = []

        # Calculate scores and prepare results for display
        for part_number, part in enumerate(parts, start=1):
            score = sum(row[4] for row in part)  # Sum of best_similarity
            # Fetch created_at from the first row in the part for display
            created_at = part[0][7] if part else "N/A"  # Assuming created_at is in the 8th column
            results.append((position, part_number, score, created_at))

        all_results[position] = (results, parts)

    return render_template('graph.html', all_results=all_results)

# Route to download PDF for a specific part
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

if __name__ == '__main__':
    app.run(debug=True)
