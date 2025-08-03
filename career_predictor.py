import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('career_data_large.csv')

# Split and clean skills column
df['Skills'] = df['Skills'].apply(lambda x: [skill.strip() for skill in x.split(',')])

# One-hot encode the skills
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(df['Skills']), columns=mlb.classes_)

# Label encode the Interest Area
le_interest = LabelEncoder()
df['Interest_Encoded'] = le_interest.fit_transform(df['Interest Area'])

# Combine all features
X = pd.concat([df[['10th %', '12th %', 'UG %', 'Interest_Encoded']], skills_encoded], axis=1)

# Encode the target (Career Path)
le_target = LabelEncoder()
y = le_target.fit_transform(df['Career Path'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print("🎯 Accuracy:", round(accuracy * 100, 2), "%")
# 🔮 Function to predict career path
def predict_career(new_input):
    interest_encoded = le_interest.transform([new_input['Interest Area']])[0]
    skill_vector = [1 if skill in new_input['Skills'] else 0 for skill in mlb.classes_]
    input_vector = [new_input['10th %'], new_input['12th %'], new_input['UG %'], interest_encoded] + skill_vector
    prediction = model.predict([input_vector])
    return le_target.inverse_transform(prediction)[0]

# 🧪 Example use:
new_user = {
    '10th %': 88,
    '12th %': 85,
    'UG %': 82,
    'Interest Area': 'Data',
    'Skills': ['Python', 'SQL']
}

print("🎓 Predicted Career:", predict_career(new_user))
