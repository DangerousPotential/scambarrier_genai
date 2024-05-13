from flask import Flask, render_template, request, session, current_app
from dotenv import load_dotenv
import sqlite3
import datetime
from textblob import TextBlob
from spellchecker import SpellChecker
import joblib
import pickle
import os
import re
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.chains import LLMChain
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

response_schemas = [
    ResponseSchema(name = 'Scam', description= 'Return 1 if the message is likely to be scam or Return 0 if the message is likely not a spam'),
    ResponseSchema(name = 'reason', description = "Give the reason on why the message is a scam or not a scam")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(template = "You are a Singaporean expert which knows all about scams in Singapore. Your task is to check whethere the inputted text is a scam and return the defined json instructions accordingly. This is the text: {text} \n {format_instructions}",
                        input_variables = ["text"],
                        partial_variables = {"format_instructions": format_instructions})
model = ChatOpenAI(openai_api_key = os.getenv('OPENAI_API_KEY'))
chain = prompt | model | output_parser
response = chain.invoke({"text": "Ah Long Lai Liao"})

def scam_classifier(text):
    response = chain.invoke({'text': text})
    return response


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
    #   if scam_classifier(text)['Scam'] == 1:
    #       result = "Scam!"
      determine_results = scam_classifier(text)
      result = f"Scam! Reason: {determine_results['reason']}" if determine_results["Scam"] == 1 else f"Not Scam! Reason: {determine_results['reason']}"
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