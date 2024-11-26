import pickle
from sklearn.dummy import DummyClassifier

# Step 1: Create a simple DummyClassifier
model = DummyClassifier(strategy="most_frequent")  # Always predicts the most common class

# Step 2: Train the model with dummy data
X_train = [[0], [1], [2], [3]]  # Dummy features
y_train = [0, 0, 1, 1]          # Dummy labels
model.fit(X_train, y_train)

# Step 3: Save the trained model to 'model.pkl'
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'model.pkl'")
