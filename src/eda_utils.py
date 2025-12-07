# src/eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
def load_data(filepath, sep="|"):
    """
    Load insurance dataset from a text or CSV file.
    
    Args:
        filepath (str): file path to dataset
        sep (str): separator (default = pipe '|')

    Returns:
        DataFrame
    """
    df = pd.read_csv(filepath, sep=sep)
    return df


# -----------------------------------------------------------
# 2. Basic Dataset Overview
# -----------------------------------------------------------
def dataset_overview(df):
    """
    Print general dataset information.
    """
    print("\n--- SHAPE ---")
    print(df.shape)

    print("\n--- COLUMNS ---")
    print(df.columns.tolist())

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())


# -----------------------------------------------------------
# 3. Summary Statistics
# -----------------------------------------------------------
def summary_statistics(df, save_path=None):
    """
    Generate summary stats for numeric & categorical variables.
    """
    numeric_stats = df.describe()
    categorical_stats = df.describe(include='object')

    if save_path:
        numeric_stats.to_csv(os.path.join(save_path, "numeric_summary.csv"))
        categorical_stats.to_csv(os.path.join(save_path, "categorical_summary.csv"))

    return numeric_stats, categorical_stats


# -----------------------------------------------------------
# 4. Missing Value Visualization
# -----------------------------------------------------------
def plot_missing_values(df, save_path=None):
    """
    Plot heatmap of missing values.
    """
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Value Heatmap")

    if save_path:
        plt.savefig(os.path.join(save_path, "missing_values_heatmap.png"))
    plt.show()


# -----------------------------------------------------------
# 5. Correlation Plot
# -----------------------------------------------------------
def plot_correlation(df, save_path=None):
    """
    Plot correlation heatmap for numerical columns.
    """
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    if save_path:
        plt.savefig(os.path.join(save_path, "correlation_heatmap.png"))
    plt.show()


# -----------------------------------------------------------
# 6. Distribution Plot for Numerical Features
# -----------------------------------------------------------
def plot_distribution(df, numerical_columns, save_path=None):
    """
    Plot distribution for list of numerical columns.
    """
    for col in numerical_columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")

        if save_path:
            plt.savefig(os.path.join(save_path, f"dist_{col}.png"))

        plt.show()


# -----------------------------------------------------------
# 7. Bar Plots for Categorical Variables
# -----------------------------------------------------------
def plot_categorical(df, categorical_columns, save_path=None):
    """
    Plot bar chart for categorical variables.
    """
    for col in categorical_columns:
        plt.figure(figsize=(8,4))
        df[col].value_counts().head(20).plot(kind="bar")
        plt.title(f"Counts of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

        if save_path:
            plt.savefig(os.path.join(save_path, f"bar_{col}.png"))

        plt.show()
