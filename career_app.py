import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('career_data_large.csv')

# Preprocess
df['Skills'] = df['Skills'].apply(lambda x: [skill.strip() for skill in x.split(',')])
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(df['Skills']), columns=mlb.classes_)
le_interest = LabelEncoder()
df['Interest_Encoded'] = le_interest.fit_transform(df['Interest Area'])
X = pd.concat([df[['10th %', '12th %', 'UG %', 'Interest_Encoded']], skills_encoded], axis=1)
le_target = LabelEncoder()
y = le_target.fit_transform(df['Career Path'])

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# 🎨 Streamlit App
st.title("🎓 Career Path Predictor (AI Project)")
st.write("Enter your academic info, interest, and skills to get a suggested career path.")

# User inputs
tenth = st.slider("10th Percentage", 50, 100, 80)
twelfth = st.slider("12th Percentage", 50, 100, 80)
ug = st.slider("UG Percentage", 50, 100, 80)
interest = st.selectbox("Interest Area", le_interest.classes_)
selected_skills = st.multiselect("Select your skills", mlb.classes_)

# Predict button
if st.button("Predict Career Path"):
    interest_encoded = le_interest.transform([interest])[0]
    skill_vector = [1 if skill in selected_skills else 0 for skill in mlb.classes_]
    input_vector = [tenth, twelfth, ug, interest_encoded] + skill_vector
    prediction = model.predict([input_vector])
    predicted_career = le_target.inverse_transform(prediction)[0]
    st.success(f"✅ Based on your input, the suggested career path is: **{predicted_career}**")
