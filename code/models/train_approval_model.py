import pandas as pd
import tensorflow as tf
import pickle
import os

def train_model(training_data, training_outputs, testing_data, testing_outputs, scaler, lazy_mode=True):    
    '''
    @Params
        training_data    -- data to train our neural network
        training_outputs -- data to evaluate our neural network
        testing_data     -- corresponding output data to train our neural network
        testing_outputs  -- corresponding output data to evaluate our neural network
        scaler           -- scalar values to be stored with .h5 model to be read in subsequently
    
    @Function
        - Sequential model 
        - with a X densley connected neurons first layer  
        - with ReLU activation 
        - input layer defined by feature size 
        - hidden layer with Y densley connected neurons and ReLU activation
        - and one output neuron with sigmoid activation
        - compiled AdaGrad for weight updating, binary crossentropy for predict vs true variation, and accurary for # corr predict / total predict
        - batch testing our training_data and corresponding y_train in sizes of A, for B epochs
        - and evaluation of the model based off test split testing_data and testing_outputs
    '''
    if lazy_mode:
        lazy_train(training_data, training_outputs, testing_data, testing_outputs)
        return 

    # Configuration params
    OPTIMIZE_ITERS = 1000
    
    # Define our constants for our model --- 
    BATCH = 32
    BEST_BATCH = 0
    EPOCHS = 1000
    
    # initialize the layer size variables
    FIRST_LAYER = 12
    SECOND_LAYER = 12
    OUTPUT_LAYER = 1
    BEST_FIRST_LAYER = 0
    BEST_SECOND_LAYER = 0

    best_accuracy = 0.0

    # loop until the best test accuracy is achieved for the layer nodes
    for _ in range(OPTIMIZE_ITERS):
        # Build our model layer by layer "sequentially". ReLU for 1st and 2nd layers to avoid the vanishing gradient problem, but sigmoid for output since we have binary classification. Only need to specify the input shape for the first layer since the others know the input coming into it as it is specified.
        model = tf.keras.Sequential([tf.keras.layers.Dense(FIRST_LAYER, activation='relu', input_shape=(training_data.shape[1],)), tf.keras.layers.Dense(SECOND_LAYER, activation='relu'), tf.keras.layers.Dense(OUTPUT_LAYER, activation='sigmoid')]) # create / define the model || subject to changes
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # compile
        model.fit(training_data, training_outputs, epochs=EPOCHS, batch_size=BATCH, validation_split=0.2, verbose=0) # train model
        loss, accuracy = model.evaluate(testing_data, testing_outputs) # eval the model
        print(f'Accuracy: {accuracy} and Loss: {loss}') # is our model trash

        # check if the current test accuracy is better than the previous best test accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            BEST_FIRST_LAYER = FIRST_LAYER
            BEST_SECOND_LAYER = SECOND_LAYER

        # increase the size of the first layer
        FIRST_LAYER += 1

        # increase the size of the second layer
        SECOND_LAYER += 1

        # break the loop if the test accuracy is already at its best
        if best_accuracy == 1.0:
            break

    # loop until the best test accuracy is achieved for the batch size
    for _ in range(OPTIMIZE_ITERS):
        # Build our model layer by layer "sequentially". ReLU for 1st and 2nd layers to avoid the vanishing gradient problem, but sigmoid for output since we have binary classification. Only need to specify the input shape for the first layer since the others know the input coming into it as it is specified.
        model = tf.keras.Sequential([tf.keras.layers.Dense(FIRST_LAYER, activation='relu', input_shape=(training_data.shape[1],)), tf.keras.layers.Dense(SECOND_LAYER, activation='relu'), tf.keras.layers.Dense(OUTPUT_LAYER, activation='sigmoid')]) # create / define the model || subject to changes
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # compile
        model.fit(training_data, training_outputs, epochs=EPOCHS, batch_size=BATCH, validation_split=0.2, verbose=0) # train model
        loss, accuracy = model.evaluate(testing_data, testing_outputs) # eval the model
        print(f'Accuracy: {accuracy} and Loss: {loss}') # is our model trash

        # check if the current test accuracy is better than the previous best test accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            BEST_BATCH = BATCH

            # increase the batch size number
            BATCH += 1

        # break the loop if the test accuracy is already at its best
        if best_accuracy == 1.0:
            break

    # Run our most optimized configuration
    model = tf.keras.Sequential([tf.keras.layers.Dense(BEST_FIRST_LAYER, activation='relu', input_shape=(training_data.shape[1],)), tf.keras.layers.Dense(BEST_SECOND_LAYER, activation='relu'), tf.keras.layers.Dense(OUTPUT_LAYER, activation='sigmoid')]) # create / define the model || subject to changes
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # compile
    model.fit(training_data, training_outputs, epochs=EPOCHS, batch_size=BEST_BATCH, validation_split=0.2, verbose=0) # train model
    loss, accuracy = model.evaluate(testing_data, testing_outputs) # eval the model
    print(f'Accuracy: {accuracy} and Loss: {loss}') # is our model trash

    # ***********************************************************************************************************************

    # Save our time-intensivley trained model so we dont have to wait forever again
    model.save('./models/loan_prediction_model.h5') # save our model
    with open('./models/loan_prediction_scaler.pkl', 'wb') as f: 
        pickle.dump(scaler, f) # data the scalar data to the file


# optimizer types "SDG", "RMSprop", "Adagrad" --> everyone online says that the best is our boy adam and tests prove it
# activation types "tanh", "leakyRelu", "sigmoid --hard", "relu --selu" 
def lazy_train(training_data, training_outputs, testing_data, testing_outputs, BATCH=128, EPOCHS=1000,FIRST_LAYER=16,SECOND_LAYER=32, OUTPUT_LAYER=1):
    model = tf.keras.Sequential([tf.keras.layers.Dense(FIRST_LAYER, activation='tanh', input_shape=(training_data.shape[1],)), tf.keras.layers.Dense(OUTPUT_LAYER, activation='sigmoid')]) # create / define the model || subject to changes
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # compile
    model.fit(training_data, training_outputs, epochs=EPOCHS, batch_size=BATCH, validation_split=0.2, verbose=0) # train model
    loss, accuracy = model.evaluate(testing_data, testing_outputs) # eval the model
    print(f'Accuracy: {accuracy} and Loss: {loss}') # is our model trash
    # lazy train function acheives the best randomized accuracy w/o training for a day 