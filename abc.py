import pandas as pd
from sklearn import preprocessing
import keras
from keras import layers
from keras import ops

df_train = pd.read_csv('raw182_Training_Relabeled_Auto_25.csv')
df_test = pd.read_csv('raw91_Testing_Relabeled_Auto_25.csv')
df_adl = pd.read_csv('Raw_Data_90ADL.csv')

#Separar las features
X_train = df_train.iloc[:, :3]
y_train = df_train.iloc[:, 3]

X_test = df_test.iloc[:, :3]
y_test = df_test.iloc[:, 3]

X_adl = df_adl.iloc[:, :3]
y_adl = df_adl.iloc[:, 3]

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_test_adl = scaler.fit_transform(X_adl)

model = Sequential()
model.add(Dense(64, input_dim=input_dim, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(X_test_scaled)
