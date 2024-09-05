from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

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
    user_answer = data['answer']
    correct_answer = data['correct_answer']
    similarity_score = calculate_similarity(user_answer, correct_answer)
    feedback = f"Match Score: {similarity_score}/10"
    correct = similarity_score >= 7  # Consider 7 and above as correct
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
    index = int(request.args.get('index'))
    level_df = df[df['Level'].str.lower() == level.lower()]
    if not level_df.empty:
        question = level_df.iloc[index]['question']
        answers = level_df.iloc[index]['answer']
        length = len(level_df)
        return jsonify({
            'question': question,
            'answers': answers,
            'index': index,
            'length': length
        })
    return jsonify({'error': 'No questions found for this level'})

if __name__ == '__main__':
    app.run(debug=True)
