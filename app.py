from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the vectorizer and model (relative paths, no leading slash)
vectorizer = joblib.load("bow_vectorizer.pkl")
model = joblib.load("spam_ham_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        email_text = request.form['email_text']
        # Transform input using the loaded vectorizer
        X = vectorizer.transform([email_text])
        pred = model.predict(X)[0]
        prediction = "Spam" if pred == 1 else "Ham"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)