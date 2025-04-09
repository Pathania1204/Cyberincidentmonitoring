# Cyber-Incident Feed Monitoring System

A web-based application that uses machine learning to predict cyber-incident severity levels based on network traffic parameters. The system uses an XGBoost model trained on cyber security incident data to make predictions.

## Features

- Real-time prediction of incident severity levels
- Interactive web interface with form inputs
- Visualization of severity distribution using Chart.js
- Responsive design using Tailwind CSS
- RESTful API endpoints for predictions

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cyber-incident-feed-monitoring
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the form to input incident parameters:
   - Source Port
   - Destination Port
   - Protocol
   - Packet Length
   - Anomaly Scores

4. Click "Predict Severity" to get the prediction result

## Project Structure

```
cyber-incident-feed-monitoring/
├── app.py                 # Flask application and ML model
├── requirements.txt       # Python dependencies
├── ML_Dataset.csv        # Training dataset
├── templates/
│   └── index.html        # Web interface
└── model.joblib          # Trained model (generated on first run)
```

## API Endpoints

- `GET /` - Web interface
- `POST /predict` - Make predictions
- `POST /train` - Train model and get severity distribution

## Technologies Used

- Backend:
  - Flask (Python web framework)
  - XGBoost (Machine Learning)
  - Pandas (Data processing)
  - scikit-learn (Data preprocessing)

- Frontend:
  - HTML5
  - Tailwind CSS (via CDN)
  - Chart.js (via CDN)
  - Fetch API (AJAX requests)

## Notes

- The model is automatically trained on first run
- Predictions are made in real-time
- The system uses a pre-trained model stored in `model.joblib`
- The web interface is responsive and works on both desktop and mobile devices 