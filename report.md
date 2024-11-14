# Data Exploration and Cleaning Report

## 1. Adjusting Data Summary
- **Percentage of changed data**: 0.00%
- **Percentage of removed data**: 0.00%

## 2. Data Overview

### 2.1 Data Info
The dataset consists of the following columns (with their data types):

- **rownames**: int64
- **gender**: int64
- **ethnicity**: int64
- **score**: float64
- **fcollege**: int64
- **mcollege**: int64
- **home**: int64
- **urban**: int64
- **unemp**: float64
- **wage**: float64
- **distance**: float64
- **tuition**: float64
- **education**: int64
- **income**: int64
- **region**: int64

### 2.2 Data Description
Here is a summary of the dataset's statistics for numerical columns:

            count         mean          std        min          25%          50%          75%           max
rownames   4739.0  3954.638953  5953.827761   1.000000  1185.500000  2370.000000  3554.500000  37810.000000
gender     4739.0     0.451361     0.497681   0.000000     0.000000     0.000000     1.000000      1.000000
ethnicity  4739.0     1.477738     0.762458   0.000000     1.000000     2.000000     2.000000      2.000000
score      4739.0    50.889029     8.701910  28.950001    43.924999    51.189999    57.769999     72.809998
fcollege   4739.0     0.208061     0.405963   0.000000     0.000000     0.000000     0.000000      1.000000
mcollege   4739.0     0.137371     0.344275   0.000000     0.000000     0.000000     0.000000      1.000000
home       4739.0     0.820215     0.384049   0.000000     1.000000     1.000000     1.000000      1.000000
urban      4739.0     0.232961     0.422762   0.000000     0.000000     0.000000     0.000000      1.000000
unemp      4739.0     7.597215     2.763581   1.400000     5.900000     7.100000     8.900000     24.900000
wage       4739.0     9.500506     1.343067   6.590000     8.850000     9.680000    10.150000     12.960000
distance   4739.0     1.802870     2.297128   0.000000     0.400000     1.000000     2.500000     20.000000
tuition    4739.0     0.814608     0.339504   0.257510     0.484990     0.824480     1.127020      1.404160
education  4739.0    13.807765     1.789107  12.000000    12.000000    13.000000    16.000000     18.000000
income     4739.0     0.711965     0.452895   0.000000     0.000000     1.000000     1.000000      1.000000
region     4739.0     0.198987     0.399280   0.000000     0.000000     0.000000     0.000000      1.000000

### 2.3 Missing Values Summary
The following columns had missing data, which was replaced during the cleaning process:


## 3. Visualizations
Here are some key visualizations for data analysis:

### 3.1 Distribution of Scores
![Distribution of Scores](output_images/score_distribution.png)

### 3.2 Correlation Matrix
![Correlation Matrix](output_images/correlation_matrix.png)

