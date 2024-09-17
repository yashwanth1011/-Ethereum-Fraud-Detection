import matplotlib
matplotlib.use('Agg')

from threading import Thread, Lock
import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Get the absolute path to the directory containing the script
script_directory = os.path.dirname(os.path.abspath(__file__))

pickle_file_path = os.path.join(script_directory, "pickleFile1.pkl")

model = pickle.load(open(pickle_file_path, "rb"))

plot_data = None
plot_lock = Lock()  

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html", plot=plot_data, count_0=None, count_1=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input data from the form
        float_features = [float(request.form[f]) for f in request.form]
        input_data = np.array(float_features).reshape(1, -1)

        # Normalize the input data
        normalized_data_point = normalize_data_point(input_data)

        # Make a prediction with the normalized data
        prediction = model.predict(normalized_data_point)

        # Generate the plot in the main thread (no need for a separate thread)
        count_0, count_1, plot_data = generate_plot(prediction)

        return render_template("result_1.html", prediction_text=f"The predicted value is: {prediction[0]}", plot=plot_data, count_0=count_0, count_1=count_1)
    except Exception as e:
        return render_template("result_1.html", prediction_text=f"An error occurred: {str(e)}")

def normalize_data_point(data_point):
 
    min_val = 0
    max_val = 1  

    normalized_data_point = (data_point - min_val) / (max_val - min_val)

    return normalized_data_point

@app.route('/csv_predict', methods=['POST'])
def csv_predict():
    try:
        if request.method == 'POST':
            csv_file = request.files['csvFile']
            df = pd.read_csv(csv_file)
            df = df.drop_duplicates() 
            df = df.rename(columns = {'FLAG': 'fraud_status'})
            replacing_value = {'false': '0', 'n' : '0', 'f' : '0', 'true' : '1'}
            df['fraud_status'] = df['fraud_status'].replace(replacing_value)
            categorical_columns = df.select_dtypes(include = ['object'])
            for i in categorical_columns:
                df[i] = df[i].astype('category')
            numerical_columns = df.select_dtypes(include = ['int', 'float']).columns
            for i in numerical_columns:
                df[i].fillna(df[i].mean(), inplace = True)
            for i in categorical_columns:
                df[i].fillna(df[i].mode().iloc[0], inplace = True)
            for i in numerical_columns: 
                df[i] = df[i].round(1)
            columns_to_drop = ['Unnamed: 0', 'Address', 'fraud_status']
            df = df.drop(columns=columns_to_drop)
            df.info()
            X = df.iloc[:, 0:16] 
            X = np.array(X)
            predictions = model.predict(X)
            count_0, count_1, plot_data = generate_plot(predictions)

            return render_template("result.html", prediction_text=f"The predicted value is: {predictions.tolist()}", plot=plot_data, count_0=count_0, count_1=count_1)
    except Exception as e:
        return render_template("result.html", prediction_text=f"An error occurred: {str(e)}")


def generate_plot(predictions):

    predictions_int = predictions.astype(int)

    # Count values for class 0 and class 1
    count_class_0, count_class_1 = np.bincount(predictions_int)

    # Print the count values
    print(f'Count for Class 0: {count_class_0}')
    print(f'Count for Class 1: {count_class_1}')

    # Create the plot using Agg backend
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.pie([count_class_0, count_class_1], labels=['Not Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Prediction Distribution')

    # Save the plot to a BytesIO object
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    # Return the count values and plot data
    return count_class_0, count_class_1, img_base64

if __name__ == "__main__":
    app.run(debug=True)

def generate_plot(predictions):

    predictions_int = predictions.astype(int)

    # Count values for class 0 and class 1
    count_class_0, count_class_1 = np.bincount(predictions_int)

    # Print the count values
    print(f'Count for Class 0: {count_class_0}')
    print(f'Count for Class 1: {count_class_1}')

    # Create the plot using Agg backend
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.pie([count_class_0, count_class_1], labels=['Not fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Prediction Distribution')

    # Save the plot to a BytesIO object
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    # Return the count values and plot data
    return count_class_0, count_class_1, img_base64

if __name__ == "__main__":
    app.run(debug=True)