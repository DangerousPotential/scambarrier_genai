from flask import Flask, render_template, request, session, current_app
import sqlite3
import datetime
from textblob import TextBlob
from spellchecker import SpellChecker
import joblib
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


app = Flask(__name__, template_folder='templates', static_folder='static')

# Preprocess text function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing PII (Personal Identifiable Information)
    text = re.sub(r'\b(?:\d{10,}|(?:\w+\.?)*\w+@(?:\w+\.)+[a-zA-Z]{2,}\b)', 'PII_REMOVED', text)

    # Simple Spell Checking and Correction
    corrected_text = []
    words = word_tokenize(text)
    for word in words:
        if not word.isalpha():
            continue  # Skip non-alphabetic words
        corrected_text.append(word)  # Just keep the word as is
    text = ' '.join(corrected_text)

    # Removing Special Characters and Symbols
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Joining tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text

# Load the random forest classifier
rf_classifier = joblib.load('rf_model.pkl')

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def scam_classifier(text):
    # Sample text
    sample_text = text

    # Preprocess the text
    processed_text = preprocess_text(sample_text)

    # Transform the text using the TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    # Making a prediction
    prediction = rf_classifier.predict(vectorized_text)
    if prediction[0] == [1]:
        return "Scam!"
    else:
        return "Not Scam!"

# Get the current date and time
now = datetime.datetime.now()

# Get the date and time components
current_date = now.date()
current_time = now.time()

def spchecker(text):
    spell = SpellChecker()
    words = text.split()
    error_count = 0
    for word in words:
        if not spell.correction(word) == word:
            error_count += 1
    return error_count

def tanalysis(text, sentiment, errors):
    confidence = 0
    predicteds = 0.22848046737213404 # from machinelearning model ran on codecollab
    for i in range(errors):
        confidence += i/len(text)
    more = abs(sentiment - predicteds)
    less = abs(predicteds - sentiment)
    diff = min(more, less)
    confidence+=(0.5-diff)
    return confidence

def isScam(confidence):
    if confidence > 0.6:
        return "Likely to be a scam"
    else:
        return "Not likely to be a scam"
      

conn = sqlite3.connect('data.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS messages
                  (mid INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT NOT NULL,
                  polarity REAL,
                  subjectivity REAL);''')
cursor.execute('''CREATE TABLE IF NOT EXISTS activity
                  (mid INTEGER PRIMARY KEY AUTOINCREMENT,
                  current_date DATE,
                  current_time TIME
                  ); ''')
conn.commit()
conn.close()

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def index():
  result = None
  records = None
  if request.method == 'POST':
      conn = sqlite3.connect('data.db')
      cursor = conn.cursor()
      text = request.form['box']
      analysis = TextBlob(text).sentiment
      sentiment, subjectivity = analysis.polarity, analysis.subjectivity
    
      # Convert datetime to string format
      date = now.date().strftime('%Y-%m-%d')
      time = now.time().strftime('%H:%M:%S')
      # Get the number of spelling errors in the message
      errors = spchecker(text)

      # Perform the text analysis using a machine learning model
      confidence = tanalysis(text, sentiment, errors)

      # Determine if the message is likely to be a scam or not
      result = scam_classifier(text)

      # Insert the message and its analysis results into the database
      cursor.execute('INSERT INTO messages (text, polarity, subjectivity) VALUES (?, ?, ?)', (text, sentiment, subjectivity))
      cursor.execute('INSERT INTO activity (current_date, current_time) VALUES (?,?)', (date,time))
      cursor.execute('SELECT * FROM messages;')
      cursor.execute('SELECT * FROM activity;')
      records = cursor.fetchall()
      conn.commit()
      conn.close()
      # Pass the result to the template to be displayed after the message is submitted
      return render_template('index.html', message=text, result=result, records=records, confidence=confidence)
  return render_template('index.html') 

@app.route('/usage')
def usage():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Retrieve the usage data from the activity table
    cursor.execute('SELECT current_time FROM activity')
    activity_records = cursor.fetchall()
    # Extract the hour component from the time records
    usage_hours = [int(record[0].split(':')[0]) for record in activity_records]

    # Perform analysis to determine the most frequent time range
    usage_analysis = analyze_usage(usage_hours)

    conn.close()
    return render_template('usage.html', usage_analysis=usage_analysis)

def analyze_usage(usage_hours):
    # Count the frequency of each hour
    hour_counts = {}
    for hour in usage_hours:
        hour_counts[hour] = hour_counts.get(hour, 0) + 1

    # Find the hour(s) with the maximum frequency
    max_frequency = max(hour_counts.values())
    most_frequent_hours = [hour for hour, count in hour_counts.items() if count == max_frequency]

    # Determine the time range based on the most frequent hour(s)
    time_range = determine_time_range(most_frequent_hours)

    return {'most_frequent_hours': most_frequent_hours, 'time_range': time_range}

def determine_time_range(hours):
    if len(hours) == 1:
        return f"{hours[0]}:00 - {hours[0]}:59"
    else:
        return f"{min(hours)}:00 - {max(hours)}:59"

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=81)