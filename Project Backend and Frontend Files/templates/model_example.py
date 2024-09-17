import sklearn as skl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    x = np.array(df.iloc[:, 0:16])
    y = np.array(df[target_column])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return x, y

def train_and_save_model(x_train, y_train, model_file):
    lr = LogisticRegression(solver='liblinear').fit(x_train, y_train)
    joblib.dump(lr, model_file)
    print(f"Model saved to {model_file}")

def load_and_predict(input_data, model_file):
    model = joblib.load(model_file)
    prediction = model.predict(input_data)
    print(f"Prediction: {prediction}")
    return prediction

if __name__ == "__main__":
    dataset_path = 'Data.csv'
    target_column = 'fraud_status'

    x, y = load_and_preprocess_data(dataset_path, target_column)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model_file = 'yashwanth_amrutha_phase3_1.joblib'
    train_and_save_model(x_train, y_train, model_file)
    input_data_for_prediction = x_test[0].reshape(1, -1)
    
    # Call load_and_predict to get the prediction
    prediction = load_and_predict(input_data_for_prediction, model_file)


