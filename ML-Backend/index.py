from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

app = Flask(__name__)

# Load and train PR Score Model
df_pr = pd.read_csv("./trainingDataset/pr_complexity_dataset.csv")
X_pr = df_pr[["LOC", "Commits", "Reviews", "Approvals", "Merged", "Resolved_Issues"]]
y_pr = df_pr["Complexity_Score"]
X_train_pr, X_test_pr, y_train_pr, y_test_pr = train_test_split(X_pr, y_pr, test_size=0.2, random_state=42)
pr_model = LinearRegression()
pr_model.fit(X_train_pr, y_train_pr)

# Load and train Spam Detection Model
df_spam = pd.read_csv("./trainingDataset/pr_spam_dataset.csv")
X_spam = df_spam[["LOC", "Reviews", "Approvals", "Merged", "Resolved_Issues"]]
y_spam = df_spam["Spam"]
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)
spam_model = LogisticRegression()
spam_model.fit(X_train_spam, y_train_spam)

@app.route('/predict_pr_score', methods=['POST'])
def predict_pr_score():
    data = request.get_json()
    new_pr = pd.DataFrame([data])
    predicted_score = pr_model.predict(new_pr)[0]
    return jsonify({"Complexity_Score": round(predicted_score, 2)})

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    data = request.get_json()
    new_pr = pd.DataFrame([data])
    spam_prediction = spam_model.predict(new_pr)[0]
    return jsonify({"Spam_Prediction": int(spam_prediction)})

if __name__ == '__main__':
    app.run(debug=True)
    print("Server is started")
