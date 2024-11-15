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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

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

# Function to split data into training and test sets
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and test sets.
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The column to be used as the target.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before the split.
    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """
    logging.info("Splitting data into training and test sets")
    
    X = df.drop(columns=target_column)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Data split complete: Training set size = {len(X_train)}, Test set size = {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(df, target_column, test_size=0.2, random_state=42):
    """
    Splits data, selects a model, trains it, and evaluates the model.
    Parameters:
        df (pd.DataFrame): The cleaned input DataFrame.
        target_column (str): The column used as the target.
        test_size (float): The proportion of data to include in the test set.
        random_state (int): Controls shuffling for reproducibility.
    Returns:
        model (object): Trained model object.
        train_accuracy (float): Accuracy score on training data.
        test_accuracy (float): Accuracy score on test data.
        eval_report (str): Detailed classification report on test data.
    """
    logging.info("Splitting data into training and test sets for model training.")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state)
    
    train_size = len(X_train)
    test_size = len(X_test)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logging.info("Selected Logistic Regression model for binary classification.")

    # Train the model
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Generate classification report
    eval_report = classification_report(y_test, model.predict(X_test))
    logging.info("Generated classification report for test set.")

    # Log the evaluation results
    logging.info(f"Training accuracy: {train_accuracy:.2f}")
    logging.info(f"Test accuracy: {test_accuracy:.2f}")

    return model, train_accuracy, test_accuracy, eval_report, train_size, test_size


# Function to generate the data exploration and cleaning report
def generate_report(changed_percentage, removed_percentage, missing_summary, df, model_info, output_dir="output_images"):
    """
    Generates a Markdown report summarizing data cleaning, exploration, model training, and evaluation.
    Parameters:
        changed_percentage (float): Percentage of data that was modified.
        removed_percentage (float): Percentage of data that was removed.
        missing_summary (dict): Summary of missing data replacements by column.
        df (pd.DataFrame): The cleaned DataFrame.
        model_info (dict): Dictionary with model name, training and test accuracy, and evaluation report.
        output_dir (str): Directory path for image output files.
    """
    report_content = f"""# Data Exploration and Cleaning Report

## 1. Adjusting Data Summary
- **Percentage of changed data**: {changed_percentage:.2f}%
- **Percentage of removed data**: {removed_percentage:.2f}%

## 2. Data Overview

### 2.1 Data Info
The dataset consists of the following columns with their data types:

"""
    # Add columns info in a bullet-point format
    columns_info = "\n".join([f"- **{col}**: {dtype}" for col, dtype in df.dtypes.items()])
    report_content += columns_info + "\n\n"

    # 2.2 Data Description
    report_content += """### 2.2 Data Description
Here is a summary of the dataset's statistics for numerical columns:

| Column    | Count  | Mean      | Std Dev   | Min   | 25%    | 50%    | 75%    | Max   |
|-----------|--------|-----------|-----------|-------|--------|--------|--------|-------|
"""
    # Add the data summary as a table
    numeric_desc = df.describe().T  # Transpose for readability
    for index, row in numeric_desc.iterrows():
        report_content += f"| {index} | {row['count']:.0f} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |\n"
    
    report_content += "\n"

    # 2.3 Missing Values Summary
    report_content += """### 2.3 Missing Values Summary
The following columns had missing data, which was replaced during the cleaning process:

"""
    for column, count in missing_summary.items():
        report_content += f"- **{column}**: {count} missing values replaced.\n"

    # Model Evaluation Summary
    report_content += f"""\n## 3. Model Training and Evaluation

### 3.1 Model Selection
We selected **{model_info["model_name"]}** due to:
- Its interpretability and efficiency for binary classification tasks.
- Ability to provide probability estimates, which are useful for classification tasks.

### 3.2 Data Split Summary
The data was split into training and test sets is 20% for the test set.
The data was split into:
- **Training set size**: {model_info["train_size"]}
- **Test set size**: {model_info["test_size"]}

### 3.3 Model Training Results
- **Training Accuracy**: {model_info["train_accuracy"]}%
- **Test Accuracy**: {model_info["test_accuracy"]}%

### 3.4 Model Evaluation Report
The model evaluation is summarized below:
- **Training Accuracy** is calculated based on the training data predictions.
- **Test Accuracy** is calculated based on the test data predictions.
- The **Classification Report** provides precision, recall, F1-score, and support metrics for each class. It helps assess the model's performance across different categories.

    {model_info["eval_report"]}


## 4. Visualizations
Here are some key visualizations for data analysis:

### 4.1 Distribution of Scores
![Distribution of Scores](output_images/score_distribution.png)

### 4.2 Correlation Matrix
![Correlation Matrix](output_images/correlation_matrix.png)

"""
    # Additional categorical distributions if they exist in the output directory
    for column in df.select_dtypes(include=['object']).columns:
        report_content += f"### Distribution of '{column}'\n"
        report_content += f"![{column} Distribution](output_images/{column}_distribution.png)\n"

    # Save the report to a Markdown file
    with open('report.md', 'w') as f:
        f.write(report_content)
    logging.info("Report generated and saved to report.md.")


if __name__ == "__main__":
    logging.info("Script started.")
    sheet_id = '1YkU1WJJHMv-uclbaEes4Bns2N8NM89YX-injwRmJcOQ'
    
    # Data fetching and preparation steps
    df = fetch_data(sheet_id)
    df = encode_categorical(df)
    explore_data(df)
    df_cleaned, changed_percentage, removed_percentage, missing_summary = clean_data(df)

    # Changed score to be categorical
    df_cleaned['score_category'] = pd.cut(df_cleaned['score'], bins=[0, 50, 75, 100], labels=["Low", "Medium", "High"])
    df_cleaned['score_category'] = df_cleaned['score_category'].map({"Low": 0, "Medium": 1, "High": 2})
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    logging.info("Cleaned data saved to cleaned_data.csv.")
    
    # Model training and evaluation
    model, train_accuracy, test_accuracy, eval_report, train_size, test_size = train_model(df_cleaned, target_column="score_category")

    # Model information for report
    model_info = {
        "model_name": "Logistic Regression",
        "train_accuracy": train_accuracy * 100,
        "test_accuracy": test_accuracy * 100,
        "eval_report": eval_report,
        "train_size": train_size,
        "test_size": test_size
    }

    # Generate report with model evaluation details
    generate_report(changed_percentage, removed_percentage, missing_summary, df, model_info)
    logging.info("Script finished.")

