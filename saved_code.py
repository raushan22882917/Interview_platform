from flask import Flask, render_template, request, jsonify
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

# Load the CSV file with questions and answers
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
        current_index = int(request.form.get('current_index', 0))

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
