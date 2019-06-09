'''
stacked autoencoder(Deep autoencoder) example
https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/

autoencoder pretraining example(in comment)
https://github.com/keras-team/keras/pull/371

pop example
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(num_class, activation='softmax'))
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K

rms = Adam(lr=0.0004)

def create_CNN(input_dim):
    model = Sequential()
    model.add(Conv2D(30, kernel_size = (93, 8), input_shape = (input_dim[0], input_dim[1], 1), activation = "softplus"))
    model.add(MaxPooling2D(pool_size = (1, 10)))
    model.add(Flatten())
    model.add(Dense(2, activation = "softmax"))
    model.compile(optimizer = rms, loss = "mean_squared_error", metrics = ["accuracy"])
    return model


def get_AE_weights(input_size, output_size, AE_train_data, AE_test_data, AE_train_label, AE_test_label):
    # create autoencoder
    autoencoder = Sequential()
    if output_size != 2:
        autoencoder.add(Dense(output_size, input_dim = input_size, activation = "sigmoid"))
        autoencoder.add(Dense(input_size, activation = "sigmoid"))      # no tied weight...?
        autoencoder.compile(optimizer = rms, loss = "mean_squared_error")
        autoencoder.fit(x = AE_train_data, y = AE_train_data, batch_size = 20, epochs = 200, verbose = 2, validation_data = (AE_test_data, AE_test_data))
    else:
        autoencoder.add(Dense(output_size, input_dim = input_size, activation = "softmax"))
        autoencoder.compile(optimizer = rms, loss = "mean_squared_error", metrics = ["accuracy"])
        autoencoder.fit(x = AE_train_data, y = AE_train_label, batch_size = 20, epochs = 200, verbose = 2, validation_data = (AE_test_data, AE_test_label))
    # parse encoder part
    weights = autoencoder.get_weights()[:2]
    return weights


def normalize_data(data):
    for s in range(9):
        for t in range(400):
            data[s][t] /= data[s][t].sum(axis = 0, keepdims = 1)
    return data


def load_data():
    data = []
    label = []
    for i in range(1, 10):
        data_path = "../chunk/B0" + str(i)
        label_path = data_path + "_label"
        data_path += ".npy"
        label_path += ".npy"
        temp_data = np.load(data_path)
        temp_label = np.load(label_path)
        data += [temp_data]
        label += [temp_label]
    data = np.asarray(data)
    label = np.asarray(label)
#    assert data.shape == (9, 400, 93, 37, 1)
#    assert label.shape == (9, 400, 2)
    # data.shape == (9, 400, 93, 37, 1)
    #       ㄴ> (# of subjects, # of trials(of session 1, 2, 3), # of rows, # of columns, # of channels)
    return data, label


def create_temp_model(model, input_dim):
    new_model = Sequential()
    new_model.add(Conv2D(30, kernel_size = (93, 8), input_shape = (input_dim[0], input_dim[1], 1), activation = "softplus"))
    new_model.add(Flatten())
    temp_model = Sequential()
    for i in range(2):
        temp = model.layers[i]
        temp_model.add(temp)
    new_model.set_weights(temp_model.get_weights())
    return new_model


def create_SAE(train_data, test_data, train_label, test_label):
    SAE = Sequential()
    dims = [900, 500, 200, 100, 40, 15, 5, 2]
    SAE_train_data = train_data
    SAE_test_data = test_data
    weights = []
    for i in range(7):
        # get autoencoder weight
        weights += get_AE_weights(dims[i], dims[i + 1], SAE_train_data, SAE_test_data, train_label, test_label)
        # first layer
        if i == 0:
            SAE.add(Dense(units = dims[1], input_dim = dims[0], activation = "sigmoid"))
        # last layer
        elif i == 6:
            SAE.add(Dense(units = 2, activation = "softmax"))
        # internal layer
        else:
            SAE.add(Dense(units = dims[i + 1], activation = "sigmoid"))
        SAE.set_weights(weights)
        # setting train, test data of autoencoder
        layer_output = K.function([SAE.layers[0].input], [SAE.layers[i].output])
        SAE_train_data = layer_output([train_data])[0]
        SAE_test_data = layer_output([test_data])[0]
    SAE.compile(loss = "mean_squared_error", optimizer = rms, metrics = ["accuracy"])
    return SAE


def fine_tuning(SAE, train_data, train_label, test_data, test_label):
    SAE.fit(x = train_data, y = train_label, batch_size = 40, epochs = 200, verbose = 2, validation_data = (test_data, test_label))
    return SAE


def create_final_model(CNN, SAE, input_dim):
    CNN_SAE = Sequential()
    CNN_SAE.add(Conv2D(30, kernel_size = (93, 8), input_shape = (input_dim[0], input_dim[1], 1), activation = "softplus"))
    CNN_SAE.add(Flatten())
    CNN_SAE.add(Dense(units = 500, activation = "sigmoid"))
    CNN_SAE.add(Dense(units = 200, activation = "sigmoid"))
    CNN_SAE.add(Dense(units = 100, activation = "sigmoid"))
    CNN_SAE.add(Dense(units = 40, activation = "sigmoid"))
    CNN_SAE.add(Dense(units = 15, activation = "sigmoid"))
    CNN_SAE.add(Dense(units = 5, activation = "sigmoid"))
    CNN_SAE.add(Dense(units = 2, activation = "softmax"))
    CNN_weights = CNN.get_weights()[:2]
    SAE_weights = SAE.get_weights()
    weights = CNN_weights + SAE_weights
    CNN_SAE.set_weights(weights)
    CNN_SAE.compile(loss = "mean_squared_error", optimizer = rms, metrics = ["accuracy"])
    return CNN_SAE


def asd():
    import cross_validation as cv
    # load data
    data, label = load_data() 
    # normalize data
    normalize_data(data)
    # set dimension of input data
    input_dim = (93, 37)
    for subject_num in range(9):
        # accuracy for all subject, all trials
        CNN_score = []
        CNN_SAE_score = []
        kv = cv.gen_kv_idx(y = label[subject_num], order = 10)
        # setting train, test data
        model_num = 1
        for train_idx, test_idx in kv:
            print("*****************************")
            print("*     sub " + str(subject_num + 1) + "/9  val " + str(model_num) + "/10     *")
            print("*****************************")
            # get chunked data(for cross validation)
            CNN_train_data = data[subject_num][train_idx]
            CNN_train_label = label[subject_num][train_idx]
            CNN_test_data = data[subject_num][test_idx]
            CNN_test_label = label[subject_num][test_idx]  
            # create CNN
            CNN = create_CNN(input_dim)
            # train CNN
            CNN.fit(x = CNN_train_data, y = CNN_train_label, batch_size = 50, epochs = 300, verbose = 2, validation_data = (CNN_test_data, CNN_test_label))
            # create temp model(for autoencoder input)
            temp_model = create_temp_model(CNN, input_dim)
            # get outputs of convolutional layer of CNN
            layer_output = K.function([temp_model.layers[0].input], [temp_model.layers[-1].output])
            AE_train_data = layer_output([CNN_train_data])[0]
            AE_test_data = layer_output([CNN_test_data])[0]
            # create pre-trained Stacked Autoencoder
            SAE = create_SAE(AE_train_data, AE_test_data, CNN_train_label, CNN_test_label)
            # fine tune Stacked Autoencoder
            SAE = fine_tuning(SAE, AE_train_data, CNN_train_label, AE_test_data, CNN_test_label)
            # create final model
#            CNN_SAE = create_final_model(CNN, SAE, input_dim)
            # get the CNN accuracy
            score = CNN.evaluate(x = CNN_test_data, y = CNN_test_label, verbose = 0)
            CNN_score.append(score[1] * 100)
            # get the SAE accuracy
            score = SAE.evaluate(x = AE_test_data, y = CNN_test_label, verbose = 0)
            CNN_SAE_score.append(score[1] * 100)
            # save two models
#            CNN.save("../models/sub" + str(subject_num + 1) + "/CNN" + str(model_num) + ".h5")
#            SAE.save("../models/sub" + str(subject_num + 1) + "/SAE" + str(model_num) + ".h5")
            model_num += 1
        np.save("../models/sub" + str(subject_num + 1) + "/CNN_score", CNN_score)
        np.save("../models/sub" + str(subject_num + 1) + "/CNN_SAE_score", CNN_SAE_score)


if __name__ == "__main__":
    asd()


# 할 것
# 1. cross validation 구현
# 2. autoencoder with tied weight
# 3. gdf2npy.py에서 Bx0yE.gdf는 안건드리나?
