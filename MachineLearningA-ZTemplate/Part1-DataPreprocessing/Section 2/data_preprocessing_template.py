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


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

i_train, i_test, d_train, d_test = train_test_split(independent, dependent, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""