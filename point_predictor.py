import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#loading data and exploratory analysis
dataset = pd.read_csv('all_seasons.csv')
print(f'Shape of Dataset: {dataset.shape}')
print(dataset.head())
print(dataset.columns)
print(dataset.dtypes)

#data splitting
categorical_columns = ['team_abbreviation','college', 'country', 'season']

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')
print(dataset.dtypes)

numerical_columns = ['age', 'player_height',
       'player_weight', 'gp', 'reb', 'ast', 'net_rating', 
       'oreb_pct','dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']

outputs = ['pts']

#converting categorical data to tensors and stacking
team = dataset['team_abbreviation'].cat.codes.values
college = dataset['college'].cat.codes.values
country = dataset['country'].cat.codes.values
season = dataset['season'].cat.codes.values
categorical_data = np.stack([team, college, country, season], 1)
print(categorical_data[:10])

categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
print(categorical_data[:10])

#converting numerical data into tensors and stacking
numerical_data = np.stack([dataset[col].values for col in numerical_columns], 1)
print(numerical_data)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)
print(numerical_data[:10])

#converting output into tensor
outputs = torch.tensor(dataset[outputs].values).flatten()
print(outputs[:10])

#data shape analysis
print(categorical_data.shape)
print(numerical_data.shape)
print(outputs.shape)

#num unique categorical variables
print(f'Num Unique Team Values {len(pd.unique(team))}')
print(f'Num College Values {len(pd.unique(college))}')
print(f'Num Country Values {len(pd.unique(country))}')
print(f'Num Season Values {len(pd.unique(season))}')

#embedding categorical data into vectors
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
print(categorical_embedding_sizes)

#train test data splitting
total_records = 11700
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]