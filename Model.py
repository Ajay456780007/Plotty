import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from termcolor import cprint
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from SubFunctions.Optimization import Optimization, Optimization1
from random import choice, seed
from SubFunctions.XAI import grad_cam # Importing grad cam model from xai_utils



class RecommendationNetwork:

    def __init__(self, x_train, x_test, y_train, y_test, epochs):
        # Constructor to initialize class attributes.
        self.x_train = x_train  # Training data
        self.x_test = x_test    # Testing data
        self.y_train = y_train  # Training labels
        self.y_test = y_test    # Testing labels
        self.epochs = epochs    # Number of training epochs
        self.learning_rate = 0.001
        self.batch_size = 32  # Batch size



    def CYPA(self):
        # Print a message indicating that ANN-AB  classification is being performed.
        cprint("[⚠️] CYPA ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train
        x_test = self.x_test

        # Convert binary formate
        y_train = to_categorical(self.y_train, dtype="uint8")

        input_layer = Input(shape=(x_train.shape[1]))
        x = Dense(512, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        # Output layer
        output_layer = Dense(units=y_train.shape[1], activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)
        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)

        train_model = Model(inputs=input_layer, outputs=model.layers[-3].output)

        train_feature = train_model.predict(x_train)
        test_feature = train_model.predict(x_test)

        # Create Decision Tree classifier object
        adb = AdaBoostClassifier()

        adb.fit(train_feature, self.y_train)

        # Predict the response for test dataset
        predict = adb.predict(test_feature)

        return predict

    def MMML_CRYP(self):
        # Print a message indicating that HDLNIDS classification is being performed.
        cprint("[⚠️] MMML-CRYP ", 'magenta', on_color='on_grey')
        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        # Convert binary formate
        y_train = to_categorical(self.y_train, dtype="uint8")
        # Define the architecture of the BiLSTM model.
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

        x = SimpleRNN(units=200, return_sequences=True)(input_layer)
        x = SimpleRNN(units=100, return_sequences=False)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        # Output layer
        output_layer = Dense(units=y_train.shape[1], activation='softmax')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)
        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat1 = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat1, axis=1)

        return yhat

    def BSVFM(self):
        # Print a message indicating that GJOADL-IDSN classification is being performed.
        cprint("[⚠️] BSVFM ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        # Convert binary formate
        y_train = to_categorical(self.y_train, dtype="uint8")
        # set input layer
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        # Add convolution layer with Relu Activation and max-pooling
        x = Conv1D(32, 3, padding='same')(input_layer)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = MaxPool1D(2, padding='same')(x)
        x = Conv1D(64, 3, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool1D(2, padding='same')(x)

        # Flatten the output and add fully connected layers
        x = Flatten()(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(y_train.shape[1])(x)
        output_layer = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat1 = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat1, axis=1)

        return yhat


    def IOF_LSTM(self):
        # Print a message indicating that LSTM-AE classification is being performed.
        cprint("[⚠️] IOF-LSTM ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        # Convert binary formate
        y_train = to_categorical(self.y_train, dtype="uint8")
        # set input layer
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

        x = LSTM(200, return_sequences=False)(input_layer)

        x = Activation('relu')(x)
        # output_layer = Dense()(x)
        x = Dense(y_train.shape[1])(x)
        output_layer = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat1 = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat1, axis=1)

        return yhat


    @staticmethod
    def XAI(model, x_test, y_test, y_pred, label):
        seed(0)  # For reproducibility

        target_indices = []  # List of list to store indices corresponding to each class label
        for i in range(len(np.unique(y_test))):
            target_indices.append(list(np.where(y_test == i)[0]))


        index_label = choice(target_indices[label])  # Randomly select an index for a label
        input = x_test[index_label]

        inp = np.expand_dims(input, axis=0)

        explanation = grad_cam(model, inp, 'lstm')
        # explanation = explainerIG.explain((inp, None), model, y_pred[index_label])[:, 0]

        threshold = np.sort(explanation)
        plt.figure(figsize=(24, 5))
        plt.subplot(1, 3, 1)
        plt.plot(input)
        plt.title('Input ' + str(label), weight='bold', fontsize="9")
        plt.xlabel("", weight='bold', fontsize="20")
        plt.ylabel("", weight='bold', fontsize="20")
        plt.xticks(weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)


        plt.subplot(1, 3, 2)
        plt.plot(explanation, 'r')
        plt.title('Explanation map from ' + str("Gradcam"), weight='bold', fontsize="9")
        plt.xlabel("", weight='bold', fontsize="20")
        plt.ylabel("", weight='bold', fontsize="20")
        plt.xticks(weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)

        plt.subplot(1, 3, 3)
        plt.plot((input * (explanation > threshold)), 'r')
        plt.title('Highlighted input region (by ' + str("Gradcam") + ' algorithm)', weight='bold', fontsize="9")
        plt.xlabel("", weight='bold', fontsize="20")
        plt.ylabel("", weight='bold', fontsize="20")
        plt.xticks(weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)

        plt.pause(0.5)
        plt.close()

        cprint("================================", color='blue')
        cprint(f'True Value: {label} ', color='blue')
        cprint(f'Predicted Value: {y_pred[index_label]} ', color='blue')
        cprint("================================", color='blue')



    def XAI_BMLSTM(self, epochs, opt, explain=False):
        # Print a message indicating that LSTM-AE classification is being performed.
        cprint("[⚠️] XAI-BMLSTM ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        # Convert binary formate
        y_train = to_categorical(self.y_train, dtype="uint8")
        # set input layer

        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = LSTM(units=100, return_sequences=True)(input_layer)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=100, return_sequences=True)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=64, return_sequences=True)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=64, return_sequences=False)(x)
        x = Dense(units=512)(x)
        x = Activation('relu')(x)
        x = Dense(units=256)(x)
        x = Activation('relu')(x)
        output_layer = Dense(y_train.shape[1], activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size, verbose=1, shuffle=True)

        if opt == 0:
            model = model

        else:
            model = Optimization(model, x_test, self.y_test).main_update_hyperparameters(opt)



        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)

        if explain:
            for label in range(len(np.unique(self.y_test))):
                self.XAI(model, x_test, self.y_test, yhat, label)

        return yhat


class YieldNetwork:

    def __init__(self, x_train, x_test, y_train, y_test, epochs):
        # Constructor to initialize class attributes.
        self.x_train = x_train  # Training data
        self.x_test = x_test    # Testing data
        self.y_train = y_train  # Training labels
        self.y_test = y_test    # Testing labels
        self.epochs = epochs    # Number of training epochs
        self.learning_rate = 0.001
        self.batch_size = 32  # Batch size


    def CYPA(self):
        # Print a message indicating that ANN-AB  classification is being performed.
        cprint("[⚠️] CYPA ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train
        x_test = self.x_test

        # Convert binary formate
        input_layer = Input(shape=(x_train.shape[1]))
        x = Dense(512, activation='linear')(input_layer)
        x = Dense(128, activation='linear')(x)
        x = Dense(64, activation='linear')(x)
        x = Dense(32, activation='linear')(x)

        # Output layer
        output_layer = Dense(units=1, activation='linear')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)
        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mse'])
        model.summary()

        model.fit(x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)

        train_model = Model(inputs=input_layer, outputs=model.layers[-3].output)

        train_feature = train_model.predict(x_train)
        test_feature = train_model.predict(x_test)

        # Create Decision Tree classifier object
        adb = AdaBoostRegressor()

        adb.fit(train_feature, self.y_train)

        # Predict the response for test dataset
        predict = adb.predict(test_feature)

        return predict


    def MMML_CRYP(self):
        # Print a message indicating that HDLNIDS classification is being performed.
        cprint("[⚠️] MMML-CRYP ", 'magenta', on_color='on_grey')
        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        # Define the architecture of the BiLSTM model.
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

        x = SimpleRNN(units=200, return_sequences=True)(input_layer)
        x = SimpleRNN(units=100, return_sequences=False)(x)
        x = Activation('linear')(x)
        x = Dense(32)(x)
        x = Activation('linear')(x)
        # Output layer
        output_layer = Dense(units=1, activation='linear')(x)

        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)
        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mse'])
        model.summary()

        model.fit(x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output

        return yhat.flatten()


    def BSVFM(self):
        # Print a message indicating that GJOADL-IDSN classification is being performed.
        cprint("[⚠️] BSVFM ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)


        # set input layer
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        # Add convolution layer with Relu Activation and max-pooling
        x = Conv1D(32, 3, padding='same')(input_layer)
        x = Activation('linear')(x)
        x = Dropout(0.5)(x)
        x = MaxPool1D(2, padding='same')(x)
        x = Conv1D(64, 3, padding='same')(x)
        x = Activation('linear')(x)
        x = MaxPool1D(2, padding='same')(x)

        # Flatten the output and add fully connected layers
        x = Flatten()(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        output_layer = Activation('linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mse'])
        model.summary()

        model.fit(x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output
        return yhat.flatten()


    def IOF_LSTM(self):
        # Print a message indicating that LSTM-AE classification is being performed.
        cprint("[⚠️] IOF-LSTM ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        # Convert binary formate
        # set input layer
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

        x = LSTM(200, return_sequences=False)(input_layer)

        x = Activation('linear')(x)
        # output_layer = Dense()(x)
        x = Dense(1)(x)
        output_layer = Activation('linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mse'])
        model.summary()

        model.fit(x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output

        return yhat.flatten()

    def XAI_BMLSTM(self, epochs, opt):
        # Print a message indicating that LSTM-AE classification is being performed.
        cprint("[⚠️] XAI-BMLSTM ", 'magenta', on_color='on_grey')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)


        # set input layer
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = LSTM(units=100, return_sequences=True)(input_layer)
        x = Activation('linear')(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=100, return_sequences=True)(x)
        x = Activation('linear')(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=64, return_sequences=True)(x)
        x = Activation('linear')(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=64, return_sequences=False)(x)
        x = Dense(units=512)(x)
        x = Activation('linear')(x)
        x = Dense(units=256)(x)
        x = Activation('linear')(x)
        output_layer = Dense(1, activation='linear')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['mse'])
        model.summary()

        model.fit(x_train, self.y_train, epochs=epochs, batch_size=self.batch_size, verbose=1, shuffle=True)

        if opt == 0:
            model = model

        else:
            model = Optimization1(model, x_test, self.y_test).main_update_hyperparameters(opt)

        yhat = model.predict(x_test)  # predict the model output

        return yhat.flatten()