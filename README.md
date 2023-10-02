# Movie Review Sentiment Analysis

This is a Flask application that uses a TensorFlow model to predict the sentiment of movie reviews.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Python
- Flask
- TensorFlow
- NumPy

### Installing

A step by step series of examples that tell you how to get a development environment running:

1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Run the server using `python app.py`.

## Usage

The application exposes an endpoint `/predict` which accepts POST requests. The request body should be a JSON object with a key 'reviews' containing an array of movie reviews.

Example:

```json
{
    "reviews": ["This movie was great!", "I didn't really like the film."]
}
