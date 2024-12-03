from flask import Flask, request, jsonify, render_template
import contractions
import inflect
import nltk
import pandas as pd
import pickle
import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


app = Flask(__name__)

p = inflect.engine()
lemmatizer = WordNetLemmatizer()


# Load the TF-IDF vectorizer and the prediction model
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)
    

with open('fraud_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
word_coefficients = pd.DataFrame({'word': feature_names, 'coefficient': coefficients})

def convert_numbers_to_words_inflect(text):
    return ' '.join(p.number_to_words(word) if word.isdigit() and len(word) < 10 else word for word in text.split())

def text_transform_custom(message):
    # message = remove_names(message)
    message = message.lower()
    message = re.sub(r'[\(\[].*?[\)\]]', "", message)
    message = convert_numbers_to_words_inflect(message)
    tokens = re.findall(r'\b\w+\b', message)
    y = [token for token in tokens if token.isalnum()]
    y = [contractions.fix(word) for word in y]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation and i not in ['uhhhhrmm', 'uh', 'um', 'uh-huh']]
    y = [lemmatizer.lemmatize(token) for token in y]
    return " ".join(y)

def explain_prediction(text):
    text_tfidf = tfidf_vectorizer.transform(text)
    prediction = model.predict(text_tfidf)[0]

    # Get the feature indices with non-zero values for the input text
    feature_indices = text_tfidf.nonzero()[1]
    feature_names = tfidf_vectorizer.get_feature_names_out()
    relevant_feature_names = [feature_names[i] for i in feature_indices]

    # Get the corresponding feature names and coefficients
    relevant_features = word_coefficients.loc[word_coefficients['word'].isin(relevant_feature_names)]

    # Sort by coefficient magnitude to show most influential features
    relevant_features = relevant_features.reindex(relevant_features['coefficient'].abs().sort_values(ascending=False).index)

    return prediction, relevant_features

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/api/predict'
        #    , methods=['POST']
           )
def api():
    try:
        # # Get JSON payload
        # data = request.json

        # # Validate payload
        # if 'text' not in data:
        #     return jsonify({"error": "Missing 'text' in request body"}), 400
        
        # Preprocess the text
        # raw_text = data['text']
        raw_text = "Hello, this is John from the Fraud Prevention Department at your bank. We’ve detected unauthorized activity on your account, and it’s critical that we address this immediately to secure your funds. A suspicious transaction of $2,000 was flagged, and we’ve temporarily frozen your account for your safety."
        processed_text = [text_transform_custom(raw_text)]
        result, features = explain_prediction(processed_text)
        response = {
            "text": raw_text,
            "processed_text": processed_text,
            "prediction": result,
            "relevant_words": features.tolist()
        }
        return jsonify(response)
    except Exception as e:
            # Handle errors gracefully
            return jsonify({"error": str(e)}), 500
