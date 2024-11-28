import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_data(filepath):
    """
    Load the engineered dataset.
    
    Args:
        filepath (str): Path to the engineered data file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def split_data(df, target_column):
    """
    Split the data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.
    
    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target set.
        
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.2f}")

def save_model(model, save_path):
    """
    Save the trained model to a file.
    
    Args:
        model (RandomForestClassifier): Trained model.
        save_path (str): Path to save the model.
        
    Returns:
        None
    """
    joblib.dump(model, save_path)
    print(f"Model saved at: {save_path}")

def main(data_path, model_save_path, target_column):
    """
    Main pipeline for training the model.
    
    Args:
        data_path (str): Path to the engineered dataset.
        model_save_path (str): Path to save the trained model.
        target_column (str): Name of the target column.
        
    Returns:
        None
    """
    # Load data
    df = load_data(data_path)
    print("Data Loaded. Shape:", df.shape)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

    # Train model
    model = train_model(X_train, y_train)
    print("Model trained.")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, model_save_path)

# Example usage (Update file paths and target column based on your dataset)
if __name__ == "__main__":
    engineered_data_path = "../data/processed/engineered_data.csv"
    model_path = "../models/random_forest_model.pkl"
    target_column_name = "readmitted"  # Replace with your actual target column
    
    main(engineered_data_path, model_path, target_column_name)
