import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Create Flask app
app = Flask(__name__)
model = joblib.load("yashwanth_amrutha_phase3_1.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extracting input features from the form
        float_features = [float(request.form[f]) for f in request.form]

        # Reshaping the features
        input_data = np.array(float_features).reshape(1, -1)

        # Making prediction
        prediction = model.predict(input_data)

        # Rendering the result on the index.html page
        return render_template("index.html", prediction_text=f"The predicted value is: {prediction[0]}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

