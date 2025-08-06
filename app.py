from flask import Flask, render_template, request
import pickle
from textblob import TextBlob
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

app = Flask(__name__)

# Load the spam model
with open(r'D:\GANESH\MINI PROJECTS\Flask\Ai_TooL\model\spam_model.pkl', 'rb') as file:
    spam_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/spam', methods=['GET', 'POST'])
def spam():
    result = ''
    email = ''
    if request.method == 'POST':
        email = request.form['email']
        prediction = spam_model.predict([email])[0]
        result = "📨 It's a Spam!" if prediction == 1 else "✅ Not a Spam!"
    return render_template('spam.html', result=result, email=email)

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    result = ''
    text = ''
    if request.method == 'POST':
        text = request.form['text']
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            result = "😊 Positive Sentiment"
        elif polarity < 0:
            result = "😞 Negative Sentiment"
        else:
            result = "😐 Neutral Sentiment"
    return render_template('sentiment.html', result=result, text=text)

@app.route('/summarizer',methods=['GET', 'POST'])
def text_summarizer():
    summary = ""
    text = ''
    if request.method == 'POST':
        text = request.form['text']
        if len(text.strip())>0:
            result = summarizer(text,max_length=130, min_length=30, do_sample=False)
            summary = result[0]['summary_text']
        else:
            summary = "Please enter some text to that is valid."
    return render_template('summarizer.html', summary=summary, text=text)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
