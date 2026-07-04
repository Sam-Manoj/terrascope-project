import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "datasets" / "leaf_features.csv"
    model_output_path = base_dir / "ml_pipeline" / "color_model.joblib"
    
    print("Loading tabular dataset...")
    df = pd.read_csv(csv_path)
    
    # 1. Split into Features (X) and Labels (y)
    X = df[["red", "green", "blue"]]
    y = df["label"]
    
    # 2. Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and train the Random Forest
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Test the model's accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))
    
    # 5. Save the trained model weights to a file
    joblib.dump(clf, model_output_path)
    print(f"\nModel successfully saved to {model_output_path}")

if __name__ == "__main__":
    train()