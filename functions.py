def univariate(df, sample=500):
  import seaborn as sns
  import matplotlib.pyplot as plt
  import math

  df_results = pd.DataFrame(columns=['bin_groups', 'type', 'missing', 'unique', 'min',
                                      'median', 'max', 'mode', 'mean', 'std', 'skew'])

  for col in df:
    # Features that apply to all dtypes
    dtype = df[col].dtype
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    mode = df[col].mode()[0]
    if pd.api.types.is_numeric_dtype(df[col]):
      # Features for numeric dtypes only
      min = df[col].min()
      max = df[col].max()
      mean = df[col].mean()
      median = df[col].median()
      std = df[col].std()
      skew = df[col].skew()
      df_results.loc[col] = ['-', dtype, missing, unique, min, median, max, mode,
                            round(mean, 2), round(std, 2), round(skew, 2)]
    else:
      # Features for object dtypes only
      flag = df[col].value_counts()[(df[col].value_counts() / df.shape[0]) < 0.05].shape[0]
      df_results.loc[col] = [flag, dtype, missing, unique, '-', '-', '-', mode, '-', '-', '-']

  # Make a sub-DataFrame of features that are objects or have only two values; they will need countplots
  countplots = df_results[(df_results['type']=='object') | (df_results['unique']==2)]
  # Make a sub-DataFrame of features that are floats or ints with many values which will need histograms
  histograms = df_results[(df_results['type']=='float64') | ((df_results['unique']>10) & (df_results['type']=='int64'))]
  histograms = histograms[histograms['unique']>2] # Remove those that are binary

  # Create a set of countplots for the categorical features
  f, ax = plt.subplots(1, countplots.shape[0], figsize=[countplots.shape[0] * 1.5, 1.5])
  for i, col in enumerate(countplots.index):
    g = sns.countplot(data=df, x=col, color='g', ax=ax[i]);
    g.set_yticklabels('')
    g.set_ylabel('')
    ax[i].tick_params(labelrotation=90, left=False)
    ax[i].xaxis.set_label_position('top')
    sns.despine(left=True, top=True, right=True)

  plt.subplots_adjust(hspace=2, wspace=.5)
  plt.show()

  # Create a set of histograms for the numeric features
  f, ax = plt.subplots(1, histograms.shape[0], figsize=[histograms.shape[0] * 1.5, 1.5])
  for i, col in enumerate(histograms.index):
    g = sns.histplot(data=df.sample(n=sample, random_state=1), x=col, color='b', ax=ax[i], kde=True);
    g.set_yticklabels(labels=[])
    g.set_ylabel('')
    ax[i].tick_params(left=False)
    sns.despine(left=True, top=True, right=True)

  plt.subplots_adjust(hspace=2, wspace=.5)
  plt.show()

  return df_results

# EDA functions
def univariate_stats(df, roundto=4):
  import pandas as pd
  import numpy as np
  
  df_results = pd.DataFrame(columns=['dtype','count', 'missing','unique','mode', 
                                    'min','q1','median','q3','max','mean','std',
                                    'skew','kurt'])
  
  for col in df:
    dtype = df[col].dtype
    count = df[col].count()
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    try:
      mode = df[col].mode()[0]
    except:
      print(f"Mode cannot be determined for {col}")
      mode = np.nan

    if pd.api.types.is_numeric_dtype(df[col]):
      min = df[col].min()
      q1 = df[col].quantile(0.25)
      median = df[col].median()
      q3 = df[col].quantile(0.75)
      max = df[col].max()
      mean = df[col].mean()
      std = df[col].std()
      skew = df[col].skew()
      kurt = df[col].kurt()
      
      df_results.loc[col] = [dtype,count,missing,unique,mode,round(min, roundto),round(q1, roundto),
                            round(median, roundto),round(q3, roundto),round(max, roundto),
                            round(mean, roundto),round(std, roundto),round(skew, roundto),round(kurt, roundto)]

    else:
      df_results.loc[col] = [dtype,count,missing,unique,mode,"","",
                            "","","","","","",""]
  return df_results

# cleaning funtions
def basic_wrangling(df, messages=True):
  import pandas as pd

  for col in df:
    # Drop any column that has all missing values
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    count = df[col].count()

    # Drop any column that has all unique values; unless it's a float64
    if missing == df.shape[0]:
      df.drop(columns=[col], inplace=True)
      if messages: print(f"All values missing; {col} dropped")
    # Drop any column that has all the same single value
    elif unique == count and 'float' not in str(df[col].dtype):
      df.drop(columns=[col], inplace=True)
      if messages: print(f"All values unique; {col} dropped")
    # Print out a nice message when a column is dropped so the caller knows what happened
    elif unique == 1:
      df.drop(columns=[col], inplace=True)
      if messages: print(f"Only one value; {col} dropped")

  return df

def parse_dates(df, features=[]):
  import pandas as pd
  from datetime import datetime

  for feat in features:
    if feat in df.columns:
      df[feat] = pd.to_datetime(df[feat])

      df[f'{feat}_year'] = df[feat].dt.year
      df[f'{feat}_month'] = df[feat].dt.month
      df[f'{feat}_day'] = df[feat].dt.day
      df[f'{feat}_weekday'] = df[feat].dt.day_name()

      df[f'{feat}_days_since'] = (datetime.today() - df[feat]).dt.days
  else:
      print(f'{feat} not found in DataFrame')

  return df


def skew_correct(df, feature, max_power=50, messages=True):
  import pandas as pd, numpy as np
  import seaborn as sns, matplotlib.pyplot as plt

  if not pd.api.types.is_numeric_dtype(df[feature]):
    if messages: print(f'{feature} is not numeric. No transformation performed')
    return df

  # Address missing data
  df = basic_wrangling(df, messages=False)
  if messages: print(f"{df.shape[0] - df.dropna().shape[0]} rows were dropped first due to missing data")
  df.dropna(inplace=True)

  # In case the dataset is too big, we can reduce to a subsample
  df_temp = df.copy()
  if df_temp.memory_usage().sum() > 1000000:
    df_temp = df.sample(frac=round(5000 / df.shape[0], 2))

  # Identify the proper transformation (i)
  i = 1
  skew = df_temp[feature].skew()
  if messages: print(f'Starting skew:\t{round(skew, 5)}')
  while round(skew, 2) != 0 and i <= max_power:
    i += 0.01
    if skew > 0:
      skew = np.power(df_temp[feature], 1/i).skew()
    else:
      skew = np.power(df_temp[feature], i).skew()
  if messages: print(f'Final skew:\t{round(skew, 5)} based on raising to {round(i, 2)}')

  # Make the transformed version of the feature in the df DataFrame
  if skew > -0.1 and skew < 0.1:
    if skew > 0:
      corrected = np.power(df[feature], 1/round(i, 3))
      name = f'{feature}_1/{round(i, 3)}'
    else:
      corrected = np.power(df[feature], round(i, 3))
      name = f'{feature}_{round(i, 3)}'
    df[name] = corrected  # Add the corrected version of the feature back into the original df
  else:
    name = f'{feature}_binary'
    df[name] = df[feature]
    if skew > 0:
      df.loc[df[name] == df[name].value_counts().index[0], name] = 0
      df.loc[df[name] != df[name].value_counts().index[0], name] = 1
    else:
      df.loc[df[name] == df[name].value_counts().index[0], name] = 1
      df.loc[df[name] != df[name].value_counts().index[0], name] = 0
    if messages:
      print(f'The feature {feature} could not be transformed into a normal distribution.')
      print(f'Instead, it has been converted to a binary (0/1)')

  if messages:
    f, axes = plt.subplots(1, 2, figsize=[7, 3.5])
    sns.despine(left=True)
    sns.histplot(df_temp[feature], color='b', ax=axes[0], kde=True)
    if skew > -0.1 and skew < 0.1:
      if skew > 0 :
        corrected = np.power(df_temp[feature], 1/round(i, 3))
      else:
        corrected = np.power(df_temp[feature], round(i, 3))
      df_temp['corrected'] = corrected
      sns.histplot(df_temp.corrected, color='g', ax=axes[1], kde=True)
    else:
      df_temp['corrected'] = df[feature]
      if skew > 0:
        df_temp.loc[df_temp['corrected'] == df_temp['corrected'].min(), 'corrected'] = 0
        df_temp.loc[df_temp['corrected'] > df_temp['corrected'].min(), 'corrected'] = 1
      else:
        df_temp.loc[df_temp['corrected'] == df_temp['corrected'].max(), 'corrected'] = 1
        df_temp.loc[df_temp['corrected'] < df_temp['corrected'].max(), 'corrected'] = 0
      sns.countplot(data=df_temp, x='corrected', color='g', ax=axes[1])
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()

    return df

def missing_drop(df, label="", features=[], messages=True, row_threshold=.9, col_threshold=.5):
  import pandas as pd

  start_count = df.count().sum()

  # Drop columns that are missing
  df.dropna(axis=1, thresh=round(col_threshold * df.shape[0]), inplace=True)
  # Drop all rows that have less data than the proportion that row_threshold requires
  df.dropna(axis=0, thresh=round(row_threshold * df.shape[1]), inplace=True)
  if label != "": df.dropna(axis=0, subset=[label], inplace=True)

  def generate_missing_table():
    df_results = pd.DataFrame(columns=['Missing', 'column', 'rows'])
    for feat in df:
      missing = df[feat].isna().sum()
      if missing > 0:
        memory_col = df.drop(columns=[feat]).count().sum()
        memory_rows = df.dropna(subset=[feat]).count().sum()
        df_results.loc[feat] = [missing, memory_col, memory_rows]
    return df_results

  df_results = generate_missing_table()
  while df_results.shape[0] > 0:
    max = df_results[['column', 'rows']].max(axis=1)[0]
    max_axis = df_results.columns[df_results.isin([max]).any()][0]
    print(max, max_axis)
    df_results.sort_values(by=[max_axis], ascending=False, inplace=True)
    if messages: print('\n', df_results)
    if max_axis=='rows':
      df.dropna(axis=0, subset=[df_results.index[0]], inplace=True)
    else:
      df.drop(columns=[df_results.index[0]], inplace=True)
    df_results = generate_missing_table()

  if messages: print(f'{round(df.count().sum() / start_count * 100, 2)}% ({df.count().sum()}) / ({start_count}) of non-null cells were kept.')
  return df

def univariate_charts(df, box=True, hist=True, save=False, save_path='', stats=True):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.set(style="ticks")

  for col in df.columns:
    plt.figure(figsize=(8, 5))

    if pd.api.types.is_numeric_dtype(df[col]):
      if box and hist:
        fig, (ax_box, ax_hist) = plt.subplots(
            2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.8)}, figsize=(8, 5)
        )
        sns.boxplot(x=df[col], ax=ax_box, fliersize=4, width=0.5, linewidth=1)
        sns.histplot(df[col], kde=True, ax=ax_hist)
        ax_box.set(yticks=[], xlabel='')
        sns.despine(ax=ax_box, left=True)
        sns.despine(ax=ax_hist)
      elif box:
        sns.boxplot(x=df[col], fliersize=4, width=0.5, linewidth=1)
        sns.despine()
      elif hist:
        sns.histplot(df[col], kde=True, rug=True)
        sns.despine()

      if stats:
        stats_text = (
          f"Unique: {df[col].nunique()}\n"
          f"Missing: {df[col].isnull().sum()}\n"
          f"Mode: {df[col].mode().iloc[0]}\n"
          f"Min: {df[col].min():.2f}\n"
          f"25%: {df[col].quantile(0.25):.2f}\n"
          f"Median: {df[col].median():.2f}\n"
          f"75%: {df[col].quantile(0.75):.2f}\n"
          f"Max: {df[col].max():.2f}\n"
          f"Std dev: {df[col].std():.2f}\n"
          f"Mean: {df[col].mean():.2f}\n"
          f"Skew: {df[col].skew():.2f}\n"
          f"Kurt: {df[col].kurt():.2f}"
        )
        plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, va='center', transform=plt.gcf().transFigure)
    else:
      sns.countplot(x=col, data=df, order=df[col].value_counts().index, hue=col, dodge=False, legend=False, palette="RdBu_r")
      sns.despine()
      if stats:
        stats_text = (
          f"Unique: {df[col].nunique()}\n"
          f"Missing: {df[col].isnull().sum()}\n"
          f"Mode: {df[col].mode().iloc[0]}"
        )
        plt.gcf().text(0.95, 0.5, stats_text, fontsize=10, va='center', transform=plt.gcf().transFigure)

    plt.title(col, fontsize=14)
    if save:
      plt.savefig(f"{save_path}{col}.png", dpi=100, bbox_inches='tight')
    plt.show()


# Square root rule only
def numeric_bin(series, full_list=True, theory='all'):
  import numpy as np
  import pandas as pd

  # This is an inner function inside numeric_bin()
  # It allows us to avoid repeated code while also keep two functions together
  def updated_list(series, bins):
    size = (max(series) - min(series)) / bins
    edges = list(range(int(min(series)), int(max(series)), int(size)))
    edges.append(int(max(series)))# This is necessary because the range() function doesn't add the max value
    if not full_list:
      return edges
    else:
      new_series = []               # Create empty list to store new values
      for value in series:          # Loop through original list one-at-a-time
        for edge in edges:          # For each original list value, loop through a list of sorted-ascending edges
          if value <= edge:         # As soon as we find an edge value less than the original...
            new_series.append(edges.index(edge)) # ..., add the edge to the new list
            break                   # Break out of the loop since we found our edge
      return new_series

  # Create empty dictionary for output
  bin_dict = {}

  # This is where we choose the theory and call the inner function updated_list()
  if theory == 'all' or theory == 'sqrt':
    bins = np.sqrt(len(series))
    bin_dict.update({'sqrt (' + str(int(bins)) + ')':updated_list(series, bins)}) # Adding the number of (bins) to label
  if theory == 'all' or theory == 'sturges':
    bins = 1 + np.log2(len(series))
    bin_dict.update({'sturges (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'rice':
    bins = 2 * np.cbrt(len(series))
    bin_dict.update({'rice (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'scott':
    bins = (max(series) - min(series)) / ((3.5 * np.std(series)) / np.cbrt(len(series)))
    bin_dict.update({'scott (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'f-d':
    bins = (max(series) - min(series)) / ((2 * (np.quantile(series, 0.75) - np.quantile(series, 0.25))) / np.cbrt(len(series)))
    bin_dict.update({'freedman-diaconis (' + str(int(bins)) + ')':updated_list(series, bins)})
  if theory == 'all' or theory == 'variable':
    bins = 2 * len(series) ** (2/5)
    edges = []
    while len(edges) < bins:
      edges.append(int(np.quantile(series, (1 / bins) * len(edges))))
    edges.append(max(series))
    if not full_list:
      bin_dict.update({'variable-width (' + str(int(bins)) + ')':edges})
    else:
      new_series = []               # Create empty list to store new values
      for value in series:          # Loop through original list one-at-a-time
        for edge in edges:          # For each original list value, loop through a list of sorted-ascending edges
          if value <= edge:         # As soon as we find an edge value less than the original...
            new_series.append(edges.index(edge)) # ..., add the index of the edge in the edges list to the new list
            break                   # Break out of the loop since we found our edge
      bin_dict.update({'variable-width (' + str(int(bins)) + ')':new_series})

  if not full_list:
    # If they want edges only, we have to return a dictionary because each theory creates a different number of bins
    return bin_dict
  else:
    # If they want the full updated dataset, we can return a DataFrame because each column is the same length
    df = pd.DataFrame(bin_dict)
    return df
    
def bivariate(df, label, roundto=4):
  import pandas as pd
  from scipy import stats

  # Create an empty DataFrame to store the results
  output_df = pd.DataFrame(columns=['missing', 'p', 'r', 'τ', 'ρ', 'y = m(x) + b', 'F', 'X2', 'skew', 'unique', 'values'])

  # Iterate through each feature in the DataFrame
  for feature in df.columns:
    if feature != label:
      df_temp = df[[feature, label]].dropna()  # Remove rows with missing values
      missing = (df.shape[0] - df_temp.shape[0]) / df.shape[0]  # Calculate missing percentage
      unique = df_temp[feature].nunique()  # Count unique values

      # Bin categories for categorical variables
      if not pd.api.types.is_numeric_dtype(df_temp[feature]):
        df = bin_categories(df, feature)

      # Case 1: Both feature and label are numeric (continuous variables)
      if pd.api.types.is_numeric_dtype(df_temp[feature]) and pd.api.types.is_numeric_dtype(df_temp[label]):
        m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])  # Perform linear regression
        tau, tp = stats.kendalltau(df_temp[feature], df_temp[label])  # Kendall correlation
        rho, rp = stats.spearmanr(df_temp[feature], df_temp[label])  # Spearman correlation

        # Store results in the output DataFrame
        output_df.loc[feature] = [
            f'{missing:.2%}', round(p, roundto), round(r, roundto), round(tau, roundto),
            round(rho, roundto), f'y = {round(m, roundto)}(x) + {round(b, roundto)}', '-', '-',
            df_temp[feature].skew(), unique, '-']

        scatterplot(df_temp, feature, label, roundto)  # Generate a scatterplot for visualization

      # Case 2: Both feature and label are categorical (nominal variables)
      elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
        contingency_table = pd.crosstab(df_temp[feature], df_temp[label])  # Create contingency table
        X2, p, dof, expected = stats.chi2_contingency(contingency_table)  # Perform Chi-square test

        # Store results in the output DataFrame
        output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', '-', round(X2, roundto), '-', unique, df_temp[feature].unique()]

        crosstab(df_temp, feature, label, roundto)  # Generate a heatmap visualization

      # Case 3: One variable is numeric, and the other is categorical (ANOVA test)
      else:
        if pd.api.types.is_numeric_dtype(df_temp[feature]):
          skew = df_temp[feature].skew()  # Calculate skewness
          num = feature
          cat = label
        else:
          skew = '-'
          num = label
          cat = feature

        # Prepare data for ANOVA test
        groups = df_temp[cat].unique()
        group_lists = [df_temp[df_temp[cat] == g][num] for g in groups]  # List of groups

        results = stats.f_oneway(*group_lists)  # Perform one-way ANOVA test
        F = results[0]  # Extract F-statistic
        p = results[1]  # Extract p-value

        # Store results in the output DataFrame
        output_df.loc[feature] = [
            f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', round(F, roundto), '-', skew,
            unique, df_temp[cat].unique()]

        bar_chart(df_temp, cat, num, roundto)  # Generate a bar chart for visualization

  return output_df.sort_values(by=['p'])  # Return results sorted by p-value

def scatterplot(df, feature, label, roundto=3, linecolor='darkorange'):
  import pandas as pd
  from matplotlib import pyplot as plt
  import seaborn as sns
  from scipy import stats

  # Create a scatter plot with a regression line
  sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolor})

  # Perform linear regression to calculate regression statistics
  m, b, r, p, err = stats.linregress(df[feature], df[label])

  # Format the regression equation and statistics into a text string
  textstr  = 'Regression line:' + '\n'
  textstr += 'y  = ' + str(round(m, roundto)) + 'x + ' + str(round(b, roundto)) + '\n'
  textstr += 'r   = ' + str(round(r, roundto)) + '\n'  # Pearson correlation coefficient
  textstr += 'r²  = ' + str(round(r**2, roundto)) + '\n'  # Coefficient of determination (R-squared)
  textstr += 'p  = ' + str(round(p, roundto)) + '\n\n'  # P-value indicating significance

  # Display the regression statistics on the plot
  plt.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

  # Show the plot
  plt.show()

def bar_chart(df, feature, label, roundto=3):
  import pandas as pd
  from scipy import stats
  from matplotlib import pyplot as plt
  import seaborn as sns

  # Handle missing data: Remove rows where either feature or label has missing values
  df_temp = df[[feature, label]].dropna()

  # Create a bar chart displaying the mean of the label for each category in the feature
  sns.barplot(df_temp, x=feature, y=label)

  # Perform one-way ANOVA (F-test) to check if there is a statistically significant difference
  groups = df_temp[feature].unique()  # Get unique categories of the feature
  group_lists = []

  # Create a list of values for each group
  for g in groups:
    g_list = df_temp[df_temp[feature] == g][label]
    group_lists.append(g_list)

  results = stats.f_oneway(*group_lists)  # Conduct ANOVA test
  F = results[0]  # Extract the F-statistic
  p = results[1]  # Extract the p-value

  # Conduct pairwise t-tests with Bonferroni correction
  ttests = []  # Store significant t-test results
  for i1, g1 in enumerate(groups):
    for i2, g2 in enumerate(groups):
      if i2 > i1:  # Compare each unique pair once
        type_1 = df_temp[df_temp[feature] == g1]
        type_2 = df_temp[df_temp[feature] == g2]
        t, p = stats.ttest_ind(type_1[label], type_2[label])  # Perform independent t-test

        # Store the results
        ttests.append([str(g1) + ' - ' + str(g2), round(t, roundto), round(p, roundto)])

  # Compute Bonferroni-corrected p-value threshold
  p_threshold = 0.05 / len(ttests)

  # Create annotation text for the plot
  textstr  = '   ANOVA' + '\n'
  textstr += 'F: ' + str(round(F, roundto)) + '\n'
  textstr += 'p: ' + str(round(p, roundto)) + '\n\n'

  # Add only significant t-test results
  for ttest in ttests:
    if ttest[2] <= p_threshold:  # If p-value is below Bonferroni threshold
      if 'Sig. comparisons (Bonferroni-corrected)' not in textstr:
        textstr += 'Sig. comparisons (Bonferroni-corrected)' + '\n'
      textstr += str(ttest[0]) + ": t=" + str(ttest[1]) + ", p=" + str(ttest[2]) + '\n'

  # Display statistical results as text on the chart
  plt.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

  # Show the plot
  plt.show()

def bin_categories(df, feature, cutoff=0.05, replace_with='Other'):
  # create a list of feature values that are below the cutoff percentage
  other_list = df[feature].value_counts()[df[feature].value_counts() / len(df) < cutoff].index

  # Replace the value of any country in that list (using the .isin() method) with 'Other'
  df.loc[df[feature].isin(other_list), feature] = replace_with

  return df

def crosstab(df, feature, label, roundto=3):
  import pandas as pd
  from scipy.stats import chi2_contingency
  from matplotlib import pyplot as plt
  import seaborn as sns
  import numpy as np

  # Handle missing data: Remove rows where either feature or label has missing values
  df_temp = df[[feature, label]].dropna()

  # Bin categories if needed (consolidate rare categories into "Other")
  df_temp = bin_categories(df_temp, feature)

  # Generate the contingency table (crosstab)
  crosstab = pd.crosstab(df_temp[feature], df_temp[label])

  # Perform Chi-square test of independence
  X, p, dof, contingency_table = chi2_contingency(crosstab)

  # Format the test results into a text string
  textstr  = 'X²: ' + str(round(X, roundto)) + '\n'
  textstr += 'p = ' + str(round(p, roundto)) + '\n'
  textstr += 'dof = ' + str(dof)

  # Display the test results on the plot
  plt.text(0.9, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)

  # Convert expected frequencies to a DataFrame with rounded integer values
  ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)

  # Create a heatmap visualization of the contingency table
  sns.heatmap(ct_df, annot=True, fmt='d', cmap='coolwarm')

  # Show the heatmap
  plt.show()
