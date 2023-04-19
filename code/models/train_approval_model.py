import pandas as pd
import tensorflow as tf
import pickle

def train_model(X_train, X_test, y_train, y_test, scaler):    
    '''
    @Params
        X_train -- data to train our neural network
        X_test  -- data to evaluate our neural network
        y_train -- corresponding output data to train our neural network
        y_test  -- corresponding output data to evaluate our neural network
        scalar  -- scalar values to be stored with .h5 model to be read in subsequently
    
    @Function
        - Sequential model 
        - with a 64 densley connected neurons first layer  
        - with ReLU activation 
        - input layer defined by feature size 
        - hidden layer with 32 densley connected neurons and ReLU activation
        - and one output neuron with sigmoid activation
        - compiled AdaGrad for weight updating, binary crossentropy for predict vs true variation, and accurary for # corr predict / total predict
        - batch testing our X_train and corresponding y_train in sizes of 32, for 100 epochs
        - and evaluation of the model based off test split X_test and y_test
    '''
    model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')]) # create / define the model || subject to changes
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # compile
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2) # train model
    test_loss, test_accuracy = model.evaluate(X_test, y_test) # eval the model
    print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}') # is our model trash

    model.save('./models/loan_prediction_model.h5') # save our model
    with open('./models/loan_prediction_scaler.pkl', 'wb') as f: 
        pickle.dump(scaler, f) # data the scalar data to the file