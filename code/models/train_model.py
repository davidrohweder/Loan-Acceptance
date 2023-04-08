import pandas as pd
import tensorflow as tf
import pickle

def train_model(X_train, X_test, y_train, y_test, scaler):

    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')

    model.save('loan_prediction_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)