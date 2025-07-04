import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Step 1: Load and label data
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df['label'] = 0  # FAKE
real_df['label'] = 1  # REAL

# ✅ Step 2: Keep only necessary columns
fake_df = fake_df[['text', 'label']].dropna()
real_df = real_df[['text', 'label']].dropna()

# ✅ Step 3: Combine and clean
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df[df['label'].isin([0, 1])]
df.dropna(inplace=True)

# ✅ Debug to confirm it's clean
print("Unique labels in y:", df['label'].unique())
print("Any NaNs in y?", df['label'].isnull().sum())

# ✅ Step 4: Prepare data for training
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Step 5: Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Step 6: Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ✅ Step 7: Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully! Accuracy: {accuracy:.2f}")

# ✅ Step 8: Use the model
while True:
    user_input = input("\nEnter news text to check if it's FAKE or REAL (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    result = "REAL" if prediction == 1 else "FAKE"
    print(f"Prediction: {result}")
