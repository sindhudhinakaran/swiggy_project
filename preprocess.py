import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

def clean_and_preprocess(input_csv='swiggy.csv',
                         cleaned_csv='cleaned_data.csv',
                         encoded_csv='encoded_data.csv',
                         encoder_file='encoder.pkl'):
    # Load raw data
    df = pd.read_csv(input_csv)

    # Strip columns and drop duplicates
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    # Replace known problematic placeholders with NaN
    df = df.replace(['--', 'Too Few Ratings', 'license', 'â‚¹'], np.nan)

    # Define categorical columns for encoding (excluding 'name' to reduce dimensionality)
    categorical_cols = ['city', 'cuisine']

    # Impute missing values in categorical columns with 'Unknown'
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    # Clean and convert numeric columns
    df['cost'] = df['cost'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

    df['rating_count'] = df['rating_count'].astype(str).str.extract('(\d+)')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Impute missing numeric values with median
    for col in ['rating', 'rating_count', 'cost']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Save cleaned dataframe
    df.to_csv(cleaned_csv, index=True)

    # One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cat = encoder.fit_transform(df[categorical_cols])

    # Combine encoded categorical with numerical data
    num_data = df[['rating', 'rating_count', 'cost']].to_numpy()
    encoded_data = np.hstack([num_data, encoded_cat])

    # Save encoded dataset aligned with cleaned data index
    encoded_df = pd.DataFrame(encoded_data, index=df.index)
    encoded_df.to_csv(encoded_csv)

    # Save the encoder for future use (e.g., in Streamlit app)
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)

    print(f"Preprocessing finished: cleaned data saved to {cleaned_csv}, encoded data saved to {encoded_csv}, encoder saved to {encoder_file}")

if __name__ == '__main__':
    clean_and_preprocess()
