from flask import Flask, render_template, request, redirect, url_for
import pickle
import nltk
from nltk.corpus import stopwords
import string
import sqlite3
from datetime import datetime

# Suppress NLTK download messages
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer with error handling
try:
    with open('model/spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: 'spam_model.pkl' not found. Please ensure the model file exists in the 'model' directory.")
    exit(1)

try:
    with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    print("Error: 'vectorizer.pkl' not found. Please ensure the vectorizer file exists in the 'model' directory.")
    exit(1)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message TEXT,
                  prediction TEXT,
                  confidence REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

# Text cleaning function
def clean_text(text):
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Onboarding route
@app.route('/')
def index():
    return render_template('index.html')

# Detection route
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form['message']
        
        # Clean the input message
        cleaned_input = clean_text(user_input)
        
        # Convert the cleaned message to numerical features
        input_vector = vectorizer.transform([cleaned_input])
        
        # Make a prediction and get confidence score
        prediction = model.predict(input_vector)[0]
        confidence = model.predict_proba(input_vector)[0].max()
        
        # Map prediction to SPAM or HAM
        result = "SPAM" if prediction == 1 else "HAM"
        
        # Store prediction in the database
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions (message, prediction, confidence, timestamp)
                     VALUES (?, ?, ?, ?)''',
                  (user_input, result, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        
        # Redirect to the results page
        return redirect(url_for('result', result=result, confidence=confidence))
    
    # Render the detection template
    return render_template('detect.html')

# Result route
@app.route('/result')
def result():
    result = request.args.get('result')
    confidence = request.args.get('confidence')
    return render_template('result.html', result=result, confidence=confidence)

# History route
@app.route('/history')
def history():
    # Fetch prediction history from the database
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    predictions = c.fetchall()
    conn.close()
    
    # Render the history template
    return render_template('history.html', predictions=predictions)

# Learn route
@app.route('/learn')
def learn():
    return render_template('learn.html')

# Spam Types route
@app.route('/spam_types')
def spam_types():
    return render_template('spam_types.html')

# Run the app
if __name__ == '__main__':
    # Initialize the database
    init_db()
    
    # Enable debug mode for development
    app.run(debug=True)