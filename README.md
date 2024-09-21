
# Ethereum Fraud Detection

## Project Overview

This project aims to detect fraudulent transactions in Ethereum networks using machine learning techniques. It provides a web interface for interacting with the model and visualizing the results. The project consists of both a backend (machine learning model and API) and a frontend (web interface).

## Technology Stack

- **Python**: Backend programming language
- **Flask**: Web framework for serving the web interface
- **Machine Learning**: Fraud detection using pre-trained models stored as `.pkl` files
- **Jupyter Notebooks**: For data exploration and model development (`ProjectPhase1.ipynb` and `ProjectPhase2.ipynb`)
- **HTML/CSS**: Frontend user interface
- **Pandas, NumPy, Scikit-learn**: Machine learning and data manipulation libraries

## Project Structure

- `run.py`: The main application file that launches the web server.
- `model.py`: Contains the logic for loading and using the trained machine learning models.
- `pickleFile1.pkl`, `pickleFile2.pkl`: Pre-trained machine learning models.
- `static/` and `templates/`: Static files (CSS, JS) and HTML templates for the web interface.
- `fraud_detection_dataset.csv`, `Data.csv`: Datasets used for model training and evaluation.
- `ProjectPhase1.ipynb`, `ProjectPhase2.ipynb`: Jupyter notebooks that document the project phases.

## Installation Instructions

### Prerequisites

- Python 3.x
- Pip (Python package manager)

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Ethereum-Fraud-Detection
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project depends on the following Python packages:

```
blinker==1.7.0
certifi==2023.11.17
charset-normalizer==3.3.2
click==8.1.7
Flask==3.0.0
idna==3.6
importlib-metadata==7.0.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.3
python-dotenv==1.0.0
requests==2.31.0
urllib3==2.1.0
Werkzeug==3.0.1
zipp==3.17.0
```

## Usage

1. After installation, you can run the web server with the following command:
   ```bash
   python run.py
   ```

2. Open your browser and navigate to `http://localhost:5000` to access the web interface.

3. You can explore the fraud detection models via the interface, using the available dataset to predict whether transactions are fraudulent.
