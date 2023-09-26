from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding


app = Flask(__name__)
CORS(app)

app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the sentiment analysis model

# Tokenizer for text preprocessing


@app.route('/')
def home():
    return render_template('home.html', output='')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the movie review from the form
    

    # Tokenize and pad the input text

    # Predict the sentiment
    #load data set

    # Load the dataset
    data = pd.read_csv("IMDB Dataset.csv")

    # Preprocessing
    data['review'] = data['review'].str.replace('<.*?>', '', regex=True)  # Remove HTML tags
    data['review'] = data['review'].str.replace('[^\w\s]', '')  # Remove punctuation
    data['review'] = data['review'].str.lower()  # Convert to lowercase

    # Split the dataset into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)
    # Convert labels to numerical values
    train_labels = train_labels.apply(lambda x: 0 if x == 'negative' else 1)
    test_labels = test_labels.apply(lambda x: 0 if x == 'negative' else 1)
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    max_sequence_length = 200
    tokenizer.fit_on_texts(train_data)

    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_data)
    test_sequences = tokenizer.texts_to_sequences(test_data)

    
    # Padding sequences
    max_sequence_length = 200
    train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    # Convert labels to float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    
    model= tf.keras.models.load_model("mymodel.h5")
  

    # Threshold to classify reviews as positive or negative
    threshold = 0.4
    negation_words = ["not", "didn't", "isn't", "aren't", "no"]
    request_data = request.get_json()
    reviews = request_data.get('reviews')
    print(reviews)
    

# Predict sentiment for the movie reviews with negation handling
    movie_sequences = tokenizer.texts_to_sequences(reviews)
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
    # Set the output variable
    # output = f'The rating of the movie is {rating}'
    response_data = {'rating': rating}
    return jsonify(response_data)
    # return render_template('home.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
