import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

# visualize settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

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

def explore_data(df):
    logging.info("Exploring data.")
    
    info = df.info()
    logging.info(f"Data info:\n{info}")

    description = df.describe(include='all')
    logging.info(f"Data description:\n{description}")
    
    missing_data = df.isnull().sum()
    logging.info(f"Missing data:\n{missing_data}")
    
    # Histogram zmiennej 'score'
    plt.figure()
    sns.histplot(df['score'], kde=True)
    plt.title("Distribution of 'score'")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig("score_distribution.png")
    logging.info("Saved histogram for 'score'.")
    
    # Macierz korelacji
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    logging.info("Saved correlation matrix heatmap.")
    
    # Rozkład dla zmiennych kategorycznych
    for column in df.select_dtypes(include=['object']).columns:
        plt.figure()
        sns.countplot(x=column, data=df)
        plt.title(f"Distribution of '{column}'")
        plt.xticks(rotation=45)
        plt.savefig(f"{column}_distribution.png")
        logging.info(f"Saved distribution plot for '{column}'.")

def clean_data(df, threshold=0.7):
    logging.info("Starting data cleaning process.")
    original_size = df.shape[0]
    changed_cells = 0
    missing_summary = {}

    # Usunięcie wierszy z brakującymi wartościami (zachowanie 70% danych)
    df_cleaned = df.dropna(thresh=int(threshold * len(df.columns)))  
    removed_rows = original_size - df_cleaned.shape[0]
    logging.info(f"Removed {removed_rows} rows during cleaning.")

    # Wypełnianie braków medianą dla liczbowych i najczęściej występującą wartością dla kategorycznych
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

    # Obliczenie zmienionych danych w procentach
    changed_percentage = (changed_cells / df.size) * 100 if df.size > 0 else 0
    removed_percentage = (removed_rows / original_size) * 100 if original_size > 0 else 0

    logging.info(f"Data cleaning process completed. Changed data percentage: {changed_percentage:.2f}%, Removed data percentage: {removed_percentage:.2f}%.")

    for column, count in missing_summary.items():
        logging.info(f"Total missing values replaced in '{column}': {count}")

    return df_cleaned, changed_percentage, removed_percentage, missing_summary

def generate_report(changed_percentage, removed_percentage, missing_summary):
    report_content = f"""# Data Exploration and Cleaning Report

## Summary
- **Percentage of changed data**: {changed_percentage:.2f}%
- **Percentage of removed data**: {removed_percentage:.2f}%

## Missing Values Summary
"""
    for column, count in missing_summary.items():
        report_content += f"- **{column}**: {count} missing values replaced.\n"

    with open('report.md', 'w') as f:
        f.write(report_content)
    logging.info("Report generated and saved to report.md.")

if __name__ == "__main__":
    logging.info("Script started.")
    sheet_id = '1YkU1WJJHMv-uclbaEes4Bns2N8NM89YX-injwRmJcOQ'
    
    # Wczytywanie i eksploracja danych
    df = fetch_data(sheet_id)
    explore_data(df)
    
    # Czyszczenie danych
    df_cleaned, changed_percentage, removed_percentage, missing_summary = clean_data(df)
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    logging.info("Cleaned data saved to cleaned_data.csv.")

    # Generowanie raportu
    generate_report(changed_percentage, removed_percentage, missing_summary)
    logging.info("Script finished.")
