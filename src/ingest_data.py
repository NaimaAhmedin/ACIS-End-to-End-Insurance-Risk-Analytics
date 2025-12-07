import pandas as pd
import os

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def load_insurance_data(file_name: str) -> pd.DataFrame:
    """
    Load insurance dataset from data/raw folder.

    Parameters:
        file_name (str): CSV file located inside data/raw

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    file_path = os.path.join(RAW_DATA_PATH, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    return df


def save_processed_data(df: pd.DataFrame, output_name: str):
    """
    Save processed insurance dataset into the processed folder.
    """
    output_path = os.path.join(PROCESSED_DATA_PATH, output_name)

    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    df = load_insurance_data("insurance_claims.csv")
    save_processed_data(df, "insurance_claims_clean.csv")
