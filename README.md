# ACIS End-to-End Insurance Risk Analytics

**Project Overview**  
This project implements an end-to-end data engineering and analytics workflow for insurance risk analysis. It focuses on processing and analyzing insurance data to support risk assessment, exploratory data analysis (EDA), and data cleaning, with proper versioning using Git and DVC.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Task 1: Data Ingestion and Exploratory Data Analysis (EDA)](#task-1-data-ingestion-and-exploratory-data-analysis-eda)
3. [Task 2: Data Cleaning and Preparation](#task-2-data-cleaning-and-preparation)
4. [Data Versioning with DVC](#data-versioning-with-dvc)
5. [Environment Setup](#environment-setup)
6. [Usage](#usage)

---

## Project Structure

ACIS-End-to-End-Insurance-Risk-Analytics/
│
├─ data/
│ ├─ raw/ # Raw input files
│ │ └─ MachineLearningRating_v3.txt
│ └─ processed/ # Cleaned and processed data
│ └─ insurance_cleaned.csv
│
├─ notebooks/
│ ├─ task_1_eda.ipynb # Exploratory Data Analysis
│ └─ task_2_data_cleaning.ipynb# Data cleaning and preparation
│
├─ reports/ # Summary reports and visualizations
├─ src/ # Python scripts for ingestion & EDA
├─ requirements.txt # Python dependencies
└─ .dvc/ # DVC configuration files

---

## Task 1: Data Ingestion and Exploratory Data Analysis (EDA)

### Objective

The goal of Task 1 is to ingest raw insurance datasets and perform initial exploratory analysis to understand data distributions, detect missing values, and identify potential anomalies.

### Steps

1. **Data Ingestion**

   - Raw insurance data is located in `data/raw/`.
   - Python scripts in `src/ingest_data.py` handle loading and initial validation.
   - Ensured proper column naming, type conversion, and integrity checks.

2. **Exploratory Data Analysis**

   - Conducted descriptive statistics on numeric and categorical columns.
   - Generated visualizations:
     - Boxplots for `SumInsured`, `TotalPremium`, and `TotalClaims`.
     - Heatmaps for missing values detection.
     - Summary tables for categorical distributions.
   - Identified patterns, outliers, and missing data to guide cleaning in Task 2.

3. **Outputs**
   - `notebooks/reports/` contains summary CSV files and visualizations.
   - Key insights:
     - Some claims have missing or zero values.
     - Premium amounts have skewed distributions.
     - Categorical variables need consistent formatting.

---

## Task 2: Data Cleaning and Preparation

### Objective

Prepare a clean dataset ready for modeling by handling missing values, correcting data types, and performing consistency checks.

### Steps

1. **Missing Value Handling**

   - Imputed missing numeric values using median imputation.
   - Dropped rows with critical missing information where imputation was not meaningful.
   - Verified completeness post-imputation using summary reports.

2. **Data Type Corrections**

   - Ensured categorical columns are of type `string` or `category`.
   - Converted numeric columns to appropriate float/int types.

3. **Feature Engineering**

   - Derived new features where applicable, e.g., claim ratios (`TotalClaims/SumInsured`).
   - Normalized column names for consistency.

4. **Output**
   - Cleaned data saved as `data/processed/insurance_cleaned.csv`.
   - Visual reports updated to reflect cleaned data in `notebooks/reports/`.
   - Task 2 notebook: `notebooks/task_2_data_cleaning.ipynb`.

---

## Data Versioning with DVC

- Large files such as `insurance_cleaned.csv` are **tracked using DVC** instead of Git.
- Steps to push and retrieve data:

```bash
# Add large file to DVC
dvc add data/processed/insurance_cleaned.csv

# Track DVC metafile in Git
git add data/processed/insurance_cleaned.csv.dvc
git commit -m "Track processed CSV with DVC"
git push origin <branch>

# Push data to DVC remote storage
dvc push


This ensures Git repository remains lightweight while large datasets are versioned.

Environment Setup

Create a Python virtual environment:

python -m venv .venv


Activate the environment:

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt

Usage

Run the EDA notebook:

jupyter notebook notebooks/task_1_eda.ipynb


Run the data cleaning notebook:

jupyter notebook notebooks/task_2_data_cleaning.ipynb
```
