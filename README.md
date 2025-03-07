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

## Functions
### `univariate(df, sample=500)`
- **Description:** Computes summary statistics for each column in the dataframe.
- **Parameters:**
  - `df` (*DataFrame*): Input dataset.
  - `sample` (*int, optional*): Number of rows to sample for visualization (default is 500).
- **Returns:** DataFrame containing summary statistics.

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
