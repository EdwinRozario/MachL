# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# [row, columns]
independent = dataset.iloc[:, :-1].values
dependent = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(independent[:, 1:3])
independent[:, 1:3] = imputer.transform(independent[:, 1:3])

# Categorising data and encoding Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

independent_label_encoder = LabelEncoder()
independent[:, 0] = independent_label_encoder.fit_transform(independent[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
independent = onehotencoder.fit_transform(independent).toarray()
dependent_label_encoder = LabelEncoder()
dependent = dependent_label_encoder.fit_transform(dependent)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

i_train, i_test, d_train, d_test = train_test_split(independent, dependent, test_size = 0.2, random_state = 0)

# Data set scaling
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# 
# independent_scaler = StandardScaler()
# i_train = independent_scaler.fit_transform(i_train)
# i_test = independent_scaler.transform(i_test)
# =============================================================================

df_independent = pd.DataFrame(independent)
df_i_train = pd.DataFrame(i_train)
df_i_test = pd.DataFrame(i_test)



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""