This project was developed as part of a Datathon-style challenge.
The goal is to analyze student academic and skill-based data and build a machine learning model to predict whether a student will get placed or not.

The project focuses on Exploratory Data Analysis (EDA), feature engineering, and model building, while also generating meaningful insights that can help university placement cells make better decisions.

🎯 Problem Statement

Many students fail to get placed despite having good academic records.
This project aims to:

Identify key factors affecting student placement

Predict placement outcomes using machine learning

Provide actionable insights based on data

📂 Dataset

Records: 200 students

Format: CSV

Target Variable: placement_status (Placed / Not Placed)

Features:

CGPA

Number of internships

Number of projects

Certifications

Coding skill level

Communication skill level

Backlogs

Aptitude score

🛠️ Tech Stack

Programming Language: Python

Libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

🔍 Exploratory Data Analysis (EDA)

Key analyses performed:

Placement distribution analysis

CGPA vs placement comparison

Impact of internships and backlogs

Skill-level influence on placement

Correlation and feature importance analysis

🧠 Feature Engineering

Encoded categorical features

Created a new feature:
total_experience = internships + projects

Scaled numerical features for better model performance

🤖 Machine Learning Models

Two models were implemented and evaluated:

Model	Accuracy
Logistic Regression	~80%
Random Forest Classifier	~88–90%

✅ Random Forest performed best

📈 Key Insights

Students with higher CGPA and internships have a higher chance of placement

Backlogs negatively impact placement probability

Coding skill level is more important than certifications alone

Practical experience significantly improves placement chances

🏁 Conclusion

This project demonstrates how data analysis and machine learning can be used to solve real-world problems in education.
The model and insights can help placement cells take early action and guide students more effectively.

🚀 Future Improvements

Hyperparameter tuning

Cross-validation

Adding real-world placement datasets

Deploying the model as a web application

📎 How to Run

Clone the repository

Install required libraries

Run the Jupyter Notebook or Python script

pip install pandas numpy matplotlib seaborn scikit-learn

🙌 Author

Hemraj Niroula
Aspiring Data Scientist | Machine Learning Enthusiast
