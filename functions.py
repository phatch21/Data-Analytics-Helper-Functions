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