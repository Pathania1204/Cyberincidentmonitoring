**🛡️ Cyber-Incident Feed Monitoring System**

A Machine Learning–powered web application that predicts cyber-incident severity levels and attack types based on real-time network traffic parameters.
Built using Flask + XGBoost + Chart.js, this system analyzes traffic inputs and provides severity predictions with confidence scores and visual insights.

**🔥 Key Highlights**
🚨 Real-time cyber incident severity prediction
🧠 XGBoost model trained on cybersecurity dataset
📊 Interactive attack & severity distribution visualizations
🌐 RESTful API for model predictions
📱 Fully responsive UI using Tailwind CSS

**🧠 How It Works**
1.User inputs network traffic parameters:
Source Port
Destination Port
Protocol
Packet Length
Anomaly Score

2.Data is preprocessed using scikit-learn.

3.The trained XGBoost model predicts:
Severity Level (Low / Medium / High / Critical)
Attack Type (DDoS / Intrusion / Malware)

4.Results are displayed with:
Confidence scores
Severity bars
Chart-based distribution visualization

**🛠 Tech Stack**
🔹 Backend
Python
Flask
XGBoost
Pandas
scikit-learn
Joblib

🔹 Frontend
HTML5
Tailwind CSS
JavaScript (Fetch API)

**📁 Project Structure**
cyber-incident-feed-monitoring
├── app.py
├── requirements.txt
├── ML_Dataset.csv
├── model.joblib
├── templates/
│   └── index.html

🚀 Future Improvements

Add authentication & admin dashboard
Deploy using Docker + Cloud (AWS / Render)
Add real-time streaming data support
Improve model with feature engineering
