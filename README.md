# AI Stroke Detection & Digital Twin

## Overview

AI Stroke Detection & Digital Twin is a machine learning-based web application that predicts the risk of stroke using patient health parameters. The application integrates a Flask backend with a user-friendly web interface and includes a Digital Twin simulation to visualize how lifestyle changes can influence stroke risk over time.

---

## Features

- Machine learning-based stroke risk prediction
- Data preprocessing and feature engineering
- Interactive Flask web application
- Digital Twin simulation for lifestyle impact analysis
- Risk level classification (Low, Moderate, High, Critical)
- Explainable AI module for prediction transparency
- Interactive visualization of projected risk trends

---

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- Matplotlib

---

## Project Structure

```
AI-Stroke-Detection/
│── main.py
│── model_training.py
│── data_preprocessing.py
│── analyze_data.py
│── risk_scoring.py
│── explainability.py
│── digital_twin.py
│── imbalance_handling.py
│── templates/
│── static/
│── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/shlokasharma/AI-Stroke-Detection.git
```

Move into the project directory

```bash
cd AI-Stroke-Detection
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask application

```bash
python main.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## How It Works

1. Enter patient health information.
2. The data is preprocessed before prediction.
3. The trained machine learning model predicts stroke risk.
4. The Digital Twin module simulates future risk under different lifestyle scenarios.
5. The application displays the predicted probability, risk category, and simulation graph.

---

## Future Improvements

- Deploy the application on a cloud platform
- Improve prediction accuracy using advanced machine learning models
- Add user authentication
- Store prediction history in a database
- Support real-time health data integration

---

## Author

**Shloka Sharma**
B.Tech Computer Science Engineering (AI & ML)
VIT Chennai
