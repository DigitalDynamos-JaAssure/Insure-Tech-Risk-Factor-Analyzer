from flask import Flask, jsonify, request
import pickle
import pandas as pd

# Load the trained model from the pickle file
with open('risk_factor_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return ("hello")
# Define a route for making predictions


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Convert the data to a Pandas DataFrame
    data_df = pd.DataFrame.from_dict(data)

    # Convert categorical variables to dummy variables
    data_df = pd.get_dummies(data_df, drop_first=True)

    # Make a prediction using the model
    prediction = model.predict(data_df)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
