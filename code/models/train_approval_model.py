import pandas as pd
import tensorflow as tf
import pickle

def train_model(X_train, X_test, y_train, y_test, scaler):    
    model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')]) # create / define the model || subject to changes
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # compile
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2) # train model
    test_loss, test_accuracy = model.evaluate(X_test, y_test) # eval the model
    print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}') # is our model trash

    model.save('./models/loan_prediction_model.h5') # save our model
    with open('./models/loan_prediction_scaler.pkl', 'wb') as f: 
        pickle.dump(scaler, f) # data the scalar data to the file