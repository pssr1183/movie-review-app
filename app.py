from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS 
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np



app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the movie review from the form
    request_data = request.get_json()
    reviews = request_data.get('reviews')
    print(reviews)
    reviews = [review.strip() for review in reviews if review.strip()]
    if not reviews:
        return jsonify({'error': 'Please provide one or more valid movie reviews.'}), 400
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    max_sequence_length = 200
    model= tf.keras.models.load_model("mymodel.h5")
    with open('tokenizer_config.json', 'r', encoding='utf-8') as json_file:
        loaded_tokenizer_json = json_file.read()

    # Create a new tokenizer from the loaded JSON configuration
    loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)
# Use the loaded tokenizer for tokenization
    # Threshold to classify reviews as positive or negative
    threshold = 0.4
# Predict sentiment for the movie reviews with negation handling
    movie_sequences = loaded_tokenizer.texts_to_sequences(reviews)
    movie_padded = pad_sequences(movie_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    predictions = model.predict(movie_padded)

    # Calculate average sentiment score for the reviews
    average_sentiment = sum(predictions) / len(predictions)

    # Assign ratings based on average sentiment
    if average_sentiment >= threshold:
        rating = 5 if average_sentiment >= 0.7 else 4  # High rating for strong positive sentiment
    else:
        rating = 2 if average_sentiment >= 0.3 else 1  # Low rating for strong negative sentiment

    # Print the results
    print("Movie Reviews:")
    for review, sentiment in zip(reviews, predictions):
        sentiment_label = 'Positive' if sentiment >= threshold else 'Negative'
        print(f"Review: {review}\nSentiment: {sentiment_label}\n")
    print("Average Sentiment:", average_sentiment)
    print("Rating:", rating, "/5")
    response_data = {
        'rating': rating,
        'sentiment':sentiment_label
                     
                     }
    return jsonify(response_data)
    

if __name__ == '__main__':
    app.run(debug=True)
