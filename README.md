# Data Analytics Helper Functions

## Overview
This repository provides a set of helper functions for performing univariate and bivariate analysis on datasets. These functions assist in data cleaning, statistical analysis, and visualization, making exploratory data analysis (EDA) more efficient.

## Features
- **Univariate Analysis:** Summarizes individual variables in a dataset by computing key statistics such as missing values, unique counts, mode, min, max, mean, median, standard deviation, and skewness.
- **Bivariate Analysis (Planned):** Identify relationships between pairs of variables.
- **Visualization Support:** Uses Seaborn and Matplotlib to generate insightful visual representations.

## Installation
To use these functions, clone the repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/Data-Analytics-Helper-Functions.git
cd Data-Analytics-Helper-Functions
pip install -r requirements.txt  # If applicable
```

## Usage
Import the helper functions into your Python script or Jupyter Notebook:

```python
import pandas as pd
from functions import univariate

# Load your dataset
df = pd.read_csv("your_data.csv")

# Perform univariate analysis
univariate_results = univariate(df)
print(univariate_results)
```
# **Functions**

## **univariate(df, sample=500)**
**Description:** Computes summary statistics for each column in the dataframe and visualizes categorical and numerical distributions.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `sample` (*int, optional*): Number of rows to sample for visualization (default is 500).  
**Returns:** DataFrame containing summary statistics.

---

## **univariate_stats(df, roundto=4)**
**Description:** Computes detailed statistical summaries including quartiles, skewness, and kurtosis for each column in the dataframe.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `roundto` (*int, optional*): Number of decimal places to round results (default is 4).  
**Returns:** DataFrame containing statistical summaries.

---

## **basic_wrangling(df, messages=True)**
**Description:** Cleans the dataset by removing columns with all missing values, unique values, or a single constant value.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `messages` (*bool, optional*): If True, prints messages about dropped columns (default is True).  
**Returns:** Cleaned DataFrame.

---

## **parse_dates(df, features=[])**
**Description:** Converts specified columns into datetime format and extracts additional date-related features.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `features` (*list, optional*): List of columns to be parsed as datetime (default is empty list).  
**Returns:** DataFrame with additional date-related features.

---

## **skew_correct(df, feature, max_power=50, messages=True)**
**Description:** Attempts to normalize a skewed numerical feature using power transformations.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `feature` (*str*): Column name to transform.  
- `max_power` (*int, optional*): Maximum power to apply for transformation (default is 50).  
- `messages` (*bool, optional*): If True, prints transformation details (default is True).  
**Returns:** DataFrame with transformed feature.

---

## **missing_drop(df, label="", features=[], messages=True, row_threshold=0.9, col_threshold=0.5)**
**Description:** Drops columns and rows based on missing data thresholds.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `label` (*str, optional*): Column name to exclude from row dropping (default is "").  
- `features` (*list, optional*): List of features to check for missing values (default is empty list).  
- `messages` (*bool, optional*): If True, prints messages (default is True).  
- `row_threshold` (*float, optional*): Minimum proportion of non-null values required for a row (default is 0.9).  
- `col_threshold` (*float, optional*): Minimum proportion of non-null values required for a column (default is 0.5).  
**Returns:** Cleaned DataFrame.

---

## **univariate_charts(df, box=True, hist=True, save=False, save_path='', stats=True)**
**Description:** Generates box plots and histograms for numerical features or count plots for categorical features.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `box` (*bool, optional*): If True, creates box plots (default is True).  
- `hist` (*bool, optional*): If True, creates histograms (default is True).  
- `save` (*bool, optional*): If True, saves plots (default is False).  
- `save_path` (*str, optional*): Path to save plots if `save=True` (default is "").  
- `stats` (*bool, optional*): If True, displays descriptive statistics on plots (default is True).  
**Returns:** None (plots are displayed or saved).

---

## **numeric_bin(series, full_list=True, theory='all')**
**Description:** Bins numerical data using different binning strategies like square root rule, Sturges, Rice, and Freedman-Diaconis.  
**Parameters:**  
- `series` (*Series*): Numeric column to bin.  
- `full_list` (*bool, optional*): If True, returns transformed values instead of bin edges (default is True).  
- `theory` (*str, optional*): Binning method (`'all'`, `'sqrt'`, `'sturges'`, `'rice'`, `'scott'`, `'f-d'`, `'variable'`) (default is `'all'`).  
**Returns:** Dictionary of binned values or bin edges.

---

## **bivariate(df, label, roundto=4)**
**Description:** Computes bivariate relationships between features and a label, performing statistical tests (correlation, ANOVA, chi-square).  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `label` (*str*): Target variable name.  
- `roundto` (*int, optional*): Number of decimal places to round results (default is 4).  
**Returns:** DataFrame summarizing bivariate relationships.

---

## **scatterplot(df, feature, label, roundto=3, linecolor='darkorange')**
**Description:** Creates a scatter plot with a regression line for numerical relationships.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `feature` (*str*): Feature variable.  
- `label` (*str*): Target variable.  
- `roundto` (*int, optional*): Decimal places for statistics (default is 3).  
- `linecolor` (*str, optional*): Color of regression line (default is `'darkorange'`).  
**Returns:** None (plot is displayed).

---

## **bar_chart(df, feature, label, roundto=3)**
**Description:** Creates a bar chart for categorical vs. numerical variable relationships and performs ANOVA and t-tests.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `feature` (*str*): Categorical feature.  
- `label` (*str*): Numerical target variable.  
- `roundto` (*int, optional*): Decimal places for statistics (default is 3).  
**Returns:** None (plot is displayed).

---

## **bin_groups(df, features=[], cutoff=0.05, replace_with='Other', messages=True)**
**Description:** Bins rare categories in categorical features into an "Other" category.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `features` (*list, optional*): List of categorical columns to bin (default is all categorical columns).  
- `cutoff` (*float, optional*): Proportion threshold for binning (default is 0.05).  
- `replace_with` (*str, optional*): Label for binned categories (default is `'Other'`).  
- `messages` (*bool, optional*): If True, prints binning actions (default is True).  
**Returns:** Updated DataFrame.

---

## **crosstab(df, feature, label, roundto=3)**
**Description:** Creates a heatmap of a contingency table between two categorical features and performs a Chi-square test.  
**Parameters:**  
- `df` (*DataFrame*): Input dataset.  
- `feature` (*str*): Categorical feature.  
- `label` (*str*): Categorical target variable.  
- `roundto` (*int, optional*): Decimal places for statistics (default is 3).  
**Returns:** None (heatmap is displayed).

## Roadmap
- [ ] Add Bivariate Analysis Functionality
- [ ] Improve Visualization Capabilities
- [ ] Provide More Statistical Insights

## Acknowledgments
The core functions in this project were developed by Professor Mark Keith of BYU. This repository adapts and extends his work for educational and practical data analysis purposes.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## Contact
For questions or suggestions, reach out via [GitHub Issues](https://github.com/yourusername/Data-Analytics-Helper-Functions/issues).
