# 💧 Water Intake Predictor

This project is a web app that predicts personalized daily water intake based on user data using machine learning.
![Screenshot 2025-05-24 150119](https://github.com/user-attachments/assets/f2904703-9c77-40ff-8cf2-d7679ca909fd)


![image](https://github.com/user-attachments/assets/46edfc4e-a54d-4f98-a00a-43fe7c0ba530)

## 🚀 Features

- 🌐 **Frontend**: Built with [Streamlit](https://streamlit.io) for an interactive and clean UI.
- 🧠 Backend / Machine Learning Details
The core logic behind the water intake predictor is a machine learning regression model trained to estimate a person's ideal daily water intake (in liters) based on several lifestyle and physiological factors.

🔍 Dataset
The model was trained on a dataset fitness_tracking.csv which includes the following features:

Age

Gender

Weight (kg)

Height (cm)

Calories Burned

Active Minutes

Heart Rate (bpm)

Steps Taken

Hours Slept

Stress Level (1–10)

⚙️ Feature Engineering
To make the model more accurate and context-aware, we generated additional features:

BMI (Body Mass Index):

BMI
=
Weight (kg)
(
Height (cm)
100
)
2
BMI= 
( 
100
Height (cm)
​
 ) 
2
 
Weight (kg)
​
 
Activity Ratio:
Ratio of active minutes to sleep hours (to capture how physically active a person is in relation to rest).

Gender Encoding:
Male = 1, Female = 0

🧪 Model Architecture
We used GradientBoostingRegressor, an ensemble method that builds sequential decision trees. It was chosen because:

It's robust to outliers.

It handles non-linear relationships well.

Performs well without heavy hyperparameter tuning.

✅ Model Pipeline
We wrapped the model in a sklearn.pipeline.Pipeline which includes:

StandardScaler – to normalize input features.

GradientBoostingRegressor – the core ML model.

🧠 Training and Evaluation
Train/Test split: 80/20

Loss function: huber (robust to outliers)

Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

🔎 Inference Logic
The prediction is made by feeding user data into the model. Before inference:

BMI and Activity Ratio are computed dynamically.

The model returns the predicted water intake.

The prediction is clamped within a healthy range:

Minimum: weight × 0.03

Maximum: weight × 0.06

This ensures the prediction stays within a biologically realistic range.

## 📦 Requirements
🚀 How to Run This App Locally

📦 1. Clone the Repository

git clone https://github.com/your-username/water_intake_predictor.git
cd water-intake-predictor
📄 2. Install the Dependencies

pip install -r requirements.txt
⚠️ If requirements.txt is missing, install manually:

pip install pandas numpy scikit-learn streamlit joblib
📊 3. Train the ML Model (Optional but recommended)
Run the script to train and save the machine learning model:

python train_model.py
This will generate a file named water_model.pkl, contains the trained model.

🌐 4. Run the Web App
Use Streamlit to launch the app:

streamlit run app.py
