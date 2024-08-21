import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import ops

# Load the data
def load_data():
    # Load training and testing data
    df_train = pd.read_csv('raw182_Training_Relabeled_Auto_25.csv')
    df_test = pd.read_csv('raw91_Testing_Relabeled_Auto_25.csv')
    df_adl = pd.read_csv('Raw_Data_90ADL.csv')

    # Separate features and labels
    X_train = df_train[['ms_accelerometer_x', 'ms_accelerometer_y', 'ms_accelerometer_z']]
    y_train = df_train['outcome']

    X_test = df_test[['ms_accelerometer_x', 'ms_accelerometer_y', 'ms_accelerometer_z']]
    y_test = df_test['outcome']

    # Combine and preprocess ADL data (assuming no seizure detection needed for ADL data)
    X_adl = df_adl[['ms_accelerometer_x', 'ms_accelerometer_y', 'ms_accelerometer_z']]
    y_adl = df_adl['outcome']

    return X_train, y_train, X_test, y_test, X_adl, y_adl

def preprocess_data(X_train, X_test, X_adl):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_adl_scaled = scaler.transform(X_adl)
    return X_train_scaled, X_test_scaled, X_adl_scaled

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    X_train, y_train, X_test, y_test, X_adl, y_adl = load_data()
    X_train_scaled, X_test_scaled, X_adl_scaled = preprocess_data(X_train, X_test, X_adl)

    model = build_model(X_train_scaled.shape[1])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Optional: Evaluate on ADL data
    adl_loss, adl_accuracy = model.evaluate(X_adl_scaled, y_adl)
    print(f"ADL Loss: {adl_loss:.4f}")
    print(f"ADL Accuracy: {adl_accuracy:.4f}")

if __name__ == '__main__':
    main()
