# 📧 Email Spam/Ham Detection using Bag of Words and Naive Bayes

This project is an end-to-end implementation of an Email Spam/Ham (Not Spam) Detector using Natural Language Processing (NLP) techniques. It leverages the Bag of Words (BOW) model for feature extraction and the Naive Bayes classifier for prediction. The solution is deployed as a Flask web application for real-time inference.

---

## 🧠 What is NLP?

Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to read, understand, and derive meaning from human languages.

---

## 🧰 Bag of Words (BOW)

Bag of Words is a popular text representation technique in NLP. It converts text documents into fixed-length vectors by counting the frequency of each word (or n-gram) in the document, disregarding grammar and word order but keeping multiplicity.

- **Advantages:** Simple, effective for many tasks, works well with traditional ML models.
- **Limitations:** Ignores context and word order, can lead to large sparse matrices.

---

## 🤖 Naive Bayes Classifier

Naive Bayes is a family of probabilistic algorithms based on Bayes’ Theorem, assuming independence between features. It is widely used for text classification tasks like spam detection due to its simplicity and efficiency.

- **Why Naive Bayes?**  
  - Fast and efficient for large datasets  
  - Performs well with high-dimensional data (like BOW vectors)  
  - Robust to irrelevant features

---

## 🗂️ Project Structure

```
Email-spam-ham_using_BOW-project/
│
├── app.py                   # Flask web app
├── bow_vectorizer.pkl       # Saved CountVectorizer (BOW)
├── spam_ham_model.pkl       # Trained Naive Bayes model
├── requirements.txt         # Python dependencies
├── SMSSpamCollection        # Raw dataset (tab-separated)
└── templates/
    └── index.html           # HTML template for web app
```

---

## ⚙️ Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib
- nltk
- matplotlib
- seaborn

Install all dependencies with:


## 🚀 How to Run

1. **Train and Save Model (if not already done):**
    - Run your Jupyter notebook to preprocess data, train the model, and save the vectorizer and model:
      ```python
      import joblib
      joblib.dump(cv, "bow_vectorizer.pkl")
      joblib.dump(spam_detect_model, "spam_ham_model.pkl")
      ```

2. **Start the Flask App:**
    ```bash
    python app.py
    ```
    - Open your browser at [http://localhost:5000](http://localhost:5000)

3. **Usage:**
    - Paste any email text in the web form and click "Predict" to see if it is spam or ham.

---

## 📝 Example

**Input:**  
```
Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.
```
**Output:**  
```
Prediction: Spam
```

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Bag of Words Model - Wikipedia](https://en.wikipedia.org/wiki/Bag-of-words_model)
- [Naive Bayes Classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

---

## 🛠️ Notes

- Ensure the same preprocessing (tokenization, stopword removal, stemming/lemmatization) is used during both training and inference.
- The model may be biased if the dataset is imbalanced (more ham than spam).
- For best results, retrain the model if you update the dataset or preprocessing steps.

---

## 👨‍💻 Author

Aryan Patel  
Email: aryanpatell77462@gmail.com

---

## 📄 License

This project is for educational purposes only.
