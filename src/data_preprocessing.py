import pandas as pd
import numpy as np

# Define functions for preprocessing
def load_data(filepath):
    """
    Load raw dataset from the specified filepath.
    
    Args:
        filepath (str): Path to the raw data file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    # Fill missing numerical values with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def remove_outliers(df, columns):
    """
    Remove outliers from numerical columns using the IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of numerical columns to check for outliers.
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def preprocess_data(filepath, savepath):
    """
    Main preprocessing pipeline for the raw dataset.
    
    Args:
        filepath (str): Path to the raw data file.
        savepath (str): Path to save the processed data.
        
    Returns:
        pd.DataFrame: Cleaned and processed dataframe.
    """
    # Load data
    df = load_data(filepath)
    print("Data Loaded. Shape:", df.shape)

    # Handle missing values
    df = handle_missing_values(df)
    print("Missing values handled. Shape:", df.shape)

    # Remove outliers (Example: ['age', 'bmi'] columns)
    numerical_columns = ['age', 'bmi']  # Modify based on your dataset
    df = remove_outliers(df, numerical_columns)
    print("Outliers removed. Shape:", df.shape)

    # Save processed data
    df.to_csv(savepath, index=False)
    print("Processed data saved at:", savepath)

    return df

# Example usage (Replace file paths with actual paths)
if __name__ == "__main__":
    raw_data_path = "../data/raw/hospital_data.csv"
    processed_data_path = "../data/processed/cleaned_data.csv"
    preprocess_data(raw_data_path, processed_data_path)
