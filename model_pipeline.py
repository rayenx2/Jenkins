import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Data Preparation
def prepare_data(train_path, test_path):
    # Load the datasets
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: One or both of the file paths are incorrect.")
        return None, None

    # Combine train and test data for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Identify categorical and binary columns
    categorical_cols = combined_df.select_dtypes(include=['object']).columns
    binary_cols = []
    for col in categorical_cols:
        if combined_df[col].nunique() == 2 and \
           set(combined_df[col].unique()) == {'Yes', 'No'}:
           binary_cols.append(col)

    # One-hot encoding for categorical features (excluding binary)
    categorical_cols_to_encode = list(set(categorical_cols) - set(binary_cols))
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(combined_df[categorical_cols_to_encode])
    encoded_df = pd.DataFrame(encoded_features,   		columns=encoder.get_feature_names_out(categorical_cols_to_encode))
    combined_df = combined_df.drop(columns=categorical_cols_to_encode)
    combined_df = pd.concat([combined_df, encoded_df], axis=1)

    # Convert binary features to numerical (1/0)
    for col in binary_cols:
        combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
        
    # Separate train and test sets again
    train_data = combined_df[:len(train_df)]
    test_data = combined_df[len(train_df):]
    
    # Ensure target variable is handled correctly
    if 'Churn' in train_data.columns:
      X_train = train_data.drop('Churn', axis=1)
      y_train = train_data['Churn']
    else:
      print("Error: Target column 'churn' not found in training data")
      return None, None
    
    if 'Churn' in test_data.columns:
      X_test = test_data.drop('Churn', axis=1)
      y_test = test_data['Churn']
    else:
      X_test = test_data
      y_test = None # Indicate that there's no target variable for testing
    

    return X_train, y_train, X_test, y_test


# Model Training
def train_data(X_train, y_train, X_test=None, y_test=None):
    # Initialize and train the RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the training set
    y_train_pred = rf_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")
    
    # If test data is provided, evaluate on the test set
    if X_test is not None and y_test is not None:
      y_test_pred = rf_classifier.predict(X_test)
      test_accuracy = accuracy_score(y_test, y_test_pred)
      print(f"Test Accuracy: {test_accuracy}")
    
    return rf_classifier

# Model Evaluation
def evaluate_model(trained_model, X_test, y_test):
    y_pred = trained_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return metrics

# Save Model
def train_and_save_model(X_train, y_train, X_test=None, y_test=None, model_filename="modelRF.joblib"):

    # Debugging: Print columns before training
    print("Columns in X_train before training:", X_train.columns)

    # No need to encode 'State' again since it's already one-hot encoded

    # Train model
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(rf_classifier, model_filename)
    print(f"Model saved as {model_filename}")

    # Save feature names for consistency during prediction
    joblib.dump(list(X_train.columns), "model_features.joblib")
    print("Feature names saved as model_features.joblib")

    return rf_classifier
