import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(filepath):
    """
    Load processed data from the specified filepath.
    
    Args:
        filepath (str): Path to the processed data file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def create_new_features(df):
    """
    Create new features from the existing dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with new features.
    """
    # Example feature: Length of stay based on 'admission_date' and 'discharge_date'
    if 'admission_date' in df.columns and 'discharge_date' in df.columns:
        df['length_of_stay'] = (
            pd.to_datetime(df['discharge_date']) - pd.to_datetime(df['admission_date'])
        ).dt.days

    # Example feature: Flag for frequent readmissions
    if 'number_of_readmissions' in df.columns:
        df['frequent_readmission'] = (df['number_of_readmissions'] > 3).astype(int)
    
    return df

def encode_features(df, categorical_columns, numerical_columns):
    """
    Encode categorical features and scale numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_columns (list): List of categorical columns.
        numerical_columns (list): List of numerical columns.
        
    Returns:
        pd.DataFrame: Dataframe with encoded and scaled features.
        Pipeline: Fitted transformer pipeline for future use.
    """
    # One-hot encode categorical features
    one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    # Standard scale numerical features
    scaler = StandardScaler()
    
    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_columns),
            ('cat', one_hot, categorical_columns)
        ],
        remainder='passthrough'  # Keep other columns as-is
    )
    
    # Fit and transform the dataset
    df_transformed = preprocessor.fit_transform(df)
    feature_names = (
        numerical_columns
        + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))
    )
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)
    
    return df_transformed, preprocessor

def feature_engineering(filepath, savepath, categorical_columns, numerical_columns):
    """
    Main feature engineering pipeline.
    
    Args:
        filepath (str): Path to the processed data file.
        savepath (str): Path to save the transformed data.
        categorical_columns (list): List of categorical columns.
        numerical_columns (list): List of numerical columns.
        
    Returns:
        pd.DataFrame: Transformed dataset ready for model training.
    """
    # Load processed data
    df = load_data(filepath)
    print("Data Loaded. Shape:", df.shape)

    # Create new features
    df = create_new_features(df)
    print("New features created. Shape:", df.shape)

    # Encode and scale features
    df_transformed, preprocessor = encode_features(df, categorical_columns, numerical_columns)
    print("Features encoded and scaled. Shape:", df_transformed.shape)

    # Save the transformed data
    df_transformed.to_csv(savepath, index=False)
    print("Transformed data saved at:", savepath)

    return df_transformed, preprocessor

# Example usage (Update file paths and column lists based on your dataset)
if __name__ == "__main__":
    processed_data_path = "../data/processed/cleaned_data.csv"
    engineered_data_path = "../data/processed/engineered_data.csv"
    categorical_columns = ['gender', 'admission_type', 'diagnosis']
    numerical_columns = ['age', 'bmi', 'length_of_stay']  # Add actual columns from your dataset
    
    feature_engineering(processed_data_path, engineered_data_path, categorical_columns, numerical_columns)
