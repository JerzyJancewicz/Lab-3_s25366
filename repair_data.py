import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

# Visualize settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Create output directory for saving images if not exists
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fetch_data(sheet_id):
    logging.info("Fetching data from Google Sheets.")
    credentials_dict = json.loads(os.getenv("GOOGLE_SHEETS_CREDENTIALS"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    data = sheet.get_all_records()
    logging.info("Data fetched successfully.")
    return pd.DataFrame(data)

def encode_categorical(df):
    logging.info("Encoding categorical columns.")
    label_encoder = LabelEncoder()

    categorical_columns = ["gender", "ethnicity", "fcollege", "mcollege", "home", "urban", "income", "region"]
    
    # Apply label encoding to each categorical column
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df

def explore_data(df):
    logging.info("Exploring data.")
    
    info = df.info()
    logging.info(f"Data info:\n{info}")

    description = df.describe(include='all')
    logging.info(f"Data description:\n{description}")
    
    missing_data = df.isnull().sum()
    logging.info(f"Missing data:\n{missing_data}")
    
    # Histogram of 'score'
    plt.figure()
    sns.histplot(df['score'], kde=True)
    plt.title("Distribution of 'score'")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/score_distribution.png")
    logging.info("Saved histogram for 'score'.")
    
    # Correlation matrix
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    logging.info("Saved correlation matrix heatmap.")
    
    # Distribution for categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        plt.figure()
        sns.countplot(x=column, data=df)
        plt.title(f"Distribution of '{column}'")
        plt.xticks(rotation=45)
        plt.savefig(f"{output_dir}/{column}_distribution.png")
        logging.info(f"Saved distribution plot for '{column}'.")

def clean_data(df, threshold=0.7):
    logging.info("Starting data cleaning process.")
    original_size = df.shape[0]
    changed_cells = 0
    missing_summary = {}

    # Drop rows with missing values (keeping 70% of data)
    df_cleaned = df.dropna(thresh=int(threshold * len(df.columns)))  
    removed_rows = original_size - df_cleaned.shape[0]
    logging.info(f"Removed {removed_rows} rows during cleaning.")

    # Fill missing numeric values with the mean, and categorical with the mode
    for column in df_cleaned.select_dtypes(include=[np.number]).columns:
        num_missing = df_cleaned[column].isnull().sum()
        if num_missing > 0:
            mean_value = df_cleaned[column].mean()
            df_cleaned[column].fillna(mean_value, inplace=True)
            changed_cells += num_missing
            missing_summary[column] = num_missing
            logging.info(f"Filled {num_missing} missing values in '{column}' with mean value {mean_value:.2f}.")

    for column in df_cleaned.select_dtypes(include=['object']).columns:
        num_missing = df_cleaned[column].isnull().sum()
        if num_missing > 0:
            mode_value = df_cleaned[column].mode()[0]
            df_cleaned[column].fillna(mode_value, inplace=True)
            changed_cells += num_missing
            missing_summary[column] = num_missing
            logging.info(f"Filled {num_missing} missing values in '{column}' with mode '{mode_value}'.")

    # Calculate the changed and removed data percentages
    changed_percentage = (changed_cells / df.size) * 100 if df.size > 0 else 0
    removed_percentage = (removed_rows / original_size) * 100 if original_size > 0 else 0

    logging.info(f"Data cleaning process completed. Changed data percentage: {changed_percentage:.2f}%, Removed data percentage: {removed_percentage:.2f}%.")
    
    for column, count in missing_summary.items():
        logging.info(f"Total missing values replaced in '{column}': {count}")

    return df_cleaned, changed_percentage, removed_percentage, missing_summary

def generate_report(changed_percentage, removed_percentage, missing_summary, df, output_dir="output_images"):
    report_content = f"""# Data Exploration and Cleaning Report

## 1. Adjusting Data Summary
- **Percentage of changed data**: {changed_percentage:.2f}%
- **Percentage of removed data**: {removed_percentage:.2f}%

## 2. Data Overview

### 2.1 Data Info
The dataset consists of the following columns:

{df.info()}

### 2.2 Data Description
Here is a summary of the dataset's statistics, including the count, mean, standard deviation, min, max, and other metrics for numerical columns, as well as the count of unique values for categorical columns:

{df.describe(include='all')}

### 2.3 Missing Values Summary
The following columns had missing data which was replaced during the cleaning process:

"""
    for column, count in missing_summary.items():
        report_content += f"- **{column}**: {count} missing values replaced.\n"

    report_content += """
## 3. Visualizations
Here are some key visualizations for data analysis:

### 3.1 Distribution of Scores
![Distribution of Scores](output_images/score_distribution.png)

### 3.2 Correlation Matrix
![Correlation Matrix](output_images/correlation_matrix.png)

"""
    # Add distribution charts for categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        report_content += f"### 3.3 Distribution of '{column}'\n"
        report_content += f"![{column} Distribution](output_images/{column}_distribution.png)\n"

    with open('report.md', 'w') as f:
        f.write(report_content)
    logging.info("Report generated and saved to report.md.")

if __name__ == "__main__":
    logging.info("Script started.")
    sheet_id = '1YkU1WJJHMv-uclbaEes4Bns2N8NM89YX-injwRmJcOQ'
    
    df = fetch_data(sheet_id)
    
    df = encode_categorical(df)
    
    explore_data(df)
    
    # Clean data
    df_cleaned, changed_percentage, removed_percentage, missing_summary = clean_data(df)
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    logging.info("Cleaned data saved to cleaned_data.csv.")

    generate_report(changed_percentage, removed_percentage, missing_summary)
    logging.info("Script finished.")
