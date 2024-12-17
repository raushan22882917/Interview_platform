import os
from flask import Flask, render_template, request, jsonify, session
import psycopg2
from groq import Groq

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

HRRound