# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import pickle  # Import pickle for saving the model and vectorizer
import os  # Import os to create the model directory if it doesn't exist

# Step 2: Load the Dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Explore the Data
print("\nLabel distribution:")
print(df['label'].value_counts())

# Step 4: Preprocess the Text Data
nltk.download('stopwords')

def clean_text(text):
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply the cleaning function to the 'message' column
df['cleaned_message'] = df['message'].apply(clean_text)
print("\nCleaned messages:")
print(df['cleaned_message'].head())

# Step 5: Convert Text to Numerical Features
vectorizer = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label'].map({'spam': 1, 'ham': 0})

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Step 10: Save the Model and Vectorizer
# Create the model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the trained model
with open('model/spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the fitted vectorizer
with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully in the 'model/' folder!")

# Step 11: Test the Model with User Input
def predict_spam_or_ham(model, vectorizer):
    while True:
        # Get user input
        user_input = input("\nEnter a message to check if it's spam or ham (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        # Clean the input message
        cleaned_input = clean_text(user_input)

        # Convert the cleaned message to numerical features
        input_vector = vectorizer.transform([cleaned_input])

        # Make a prediction
        prediction = model.predict(input_vector)

        # Display the result
        if prediction[0] == 1:
            print("This message is **SPAM**.")
        else:
            print("This message is **HAM**.")

# Call the function to test the model
predict_spam_or_ham(model, vectorizer)