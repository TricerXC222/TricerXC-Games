import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([2, 4, 5, 4, 5])  # Target values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
prediction = model.predict([[6]])
print(f"Prediction: {prediction}")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model

import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('punkt') #one time download.

data = [
    ("I love this movie!", "positive"),
    ("This is a terrible product.", "negative"),
    ("The food was delicious.", "positive"),
    ("I'm feeling very sad.", "negative"),
    ("What a wonderful day!", "positive"),
    ("This is the worst experience ever.", "negative"),
]

texts = [text for text, _ in data]
labels = [label for _, label in data]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
new_text = ["This is a good example."]
new_X = vectorizer.transform(new_text)
new_prediction = model.predict(new_X)
print(f"Prediction: {new_prediction}")

model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

import tensorflow as tf
from tensorflow import keras

# Sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Create the model
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy}")

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt') #one time download.

text = "This is a simple example of NLP with Python."
tokens = word_tokenize(text)
print(f"Tokens: {tokens}")

#More complex NLP tasks require more in depth research, but this is a starting point.
website_database = {
    "positive": [
        "https://www.happify.com/",  # Happiness and well-being
        "https://www.goodnewsnetwork.org/", # Positive news
        "https://www.positivepsychology.com/", # Positive psychology resources
    ],
    "negative": [
        "https://www.betterhelp.com/", # Online therapy
        "https://www.crisistextline.org/", # Crisis support
        "https://www.mentalhealth.gov/", # Mental health information
    ],
}

def predict_sentiment_and_provide_websites(text, vectorizer, model, website_db):
    new_X = vectorizer.transform([text])
    prediction = model.predict(new_X)[0]  # Get the single prediction

    websites = website_db.get(prediction, []) #Get the websites, or empty list if the sentiment is not found.

    return prediction, websites
  import random

class AffectionateAI:
    def __init__(self, initial_affection=50):
        self.affection = initial_affection
        self.max_affection = 100
        self.min_affection = 0

    def respond(self, input_text):
        input_text = input_text.lower() #make input case insensitive.
        if "love" in input_text or "like" in input_text or "good" in input_text:
            self.affection = min(self.affection + random.randint(5, 15), self.max_affection)
            return self.generate_positive_response()
        elif "hate" in input_text or "bad" in input_text or "mean" in input_text:
            self.affection = max(self.affection - random.randint(5, 15), self.min_affection)
            return self.generate_negative_response()
        else:
            return self.generate_neutral_response()

    def generate_positive_response(self):
        responses = [
            "That makes me happy!",
            "I like that too!",
            "You're very kind.",
            "I feel good when you say that.",
            "I am glad you feel that way.",
        ]
        return random.choice(responses)

    def generate_negative_response(self):
        responses = [
            "I'm sorry to hear that.",
            "That makes me sad.",
            "I hope things get better.",
            "I don't like when you say that.",
            "That hurts my feelings.",
        ]
        return random.choice(responses)

    def generate_neutral_response(self):
        responses = [
            "Interesting.",
            "Okay.",
            "I understand.",
            "Tell me more.",
            "I see.",
        ]
        return random.choice(responses)

    def get_affection_level(self):
        return self.affection

# Example Usage:
ai = AffectionateAI()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = ai.respond(user_input)
    print("AI:", response)
    print(f"AI Affection Level: {ai.get_affection_level()}")

import tkinter as tk
from tkinter import scrolledtext
import random

class AffectionateAI:
    # (AffectionateAI class code from previous response)
    def __init__(self, initial_affection=50):
        self.affection = initial_affection
        self.max_affection = 100
        self.min_affection = 0

    def respond(self, input_text):
        input_text = input_text.lower()
        if "love" in input_text or "like" in input_text or "good" in input_text:
            self.affection = min(self.affection + random.randint(5, 15), self.max_affection)
            return self.generate_positive_response()
        elif "hate" in input_text or "bad" in input_text or "mean" in input_text:
            self.affection = max(self.affection - random.randint(5, 15), self.min_affection)
            return self.generate_negative_response()
        else:
            return self.generate_neutral_response()

    def generate_positive_response(self):
        responses = [
            "That makes me happy!",
            "I like that too!",
            "You're very kind.",
            "I feel good when you say that.",
            "I am glad you feel that way.",
        ]
        return random.choice(responses)

    def generate_negative_response(self):
        responses = [
            "I'm sorry to hear that.",
            "That makes me sad.",
            "I hope things get better.",
            "I don't like when you say that.",
            "That hurts my feelings.",
        ]
        return random.choice(responses)

    def generate_neutral_response(self):
        responses = [
            "Interesting.",
            "Okay.",
            "I understand.",
            "Tell me more.",
            "I see.",
        ]
        return random.choice(responses)

    def get_affection_level(self):
        return self.affection

class AI_Launcher:
    def __init__(self, root):
        self.root = root
        self.root.title("Affectionate AI Launcher")

        self.ai = AffectionateAI()

        self.input_label = tk.Label(root, text="Enter Text:")
        self.input_label.pack()

        self.input_text = tk.Entry(root, width=50)
        self.input_text.pack()

        self.send_button = tk.Button(root, text="Send to AI", command=self.send_to_ai)
        self.send_button.pack()

        self.output_label = tk.Label(root, text="AI Response:")
        self.output_label.pack()

        self.output_text = scrolledtext.ScrolledText(root, width=50, height=10)
        self.output_text.pack()

        self.affection_label = tk.Label(root, text="Affection Level: 50")
        self.affection_label.pack()

    def send_to_ai(self):
        user_input = self.input_text.get()
        response = self.ai.respond(user_input)
        self.output_text.insert(tk.END, "You: " + user_input + "\n")
        self.output_text.insert(tk.END, "AI: " + response + "\n")
        self.input_text.delete(0, tk.END) #Clears the input box after sending.

        self.affection_label.config(text=f"Affection Level: {self.ai.get_affection_level()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = Python_AI_Launcher(root)
    root.mainloop()
