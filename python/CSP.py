import numpy as np
import scipy.io as sio
import os
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)

# read_dir(path):
#   reads directory to get file(data) names
def read_dir(path):
    # file_list: list of file names
    file_list = os.listdir(path)
    return file_list


# read_file(path):
#   reads .mat file(data)
def read_file(path):
    mat_file = sio.loadmat(path)
    return mat_file


def read_data(features):
    # to extract train, test set in matlab 'struct'
    data = features[0,0]
    # reshape data x --> 
    #   from (# of folds, # of frequency bands, # of pairwise matrix, # of trials)
    #   to   (# of folds, # of trials, # of frequency bands, # of pairwise matrix)
    train_xt = data['train_x']
    test_xt = data['test_x']
    train_x = np.zeros((train_xt.shape[0], train_xt.shape[2], train_xt.shape[1]), dtype=float)
    test_x = np.zeros((test_xt.shape[0], test_xt.shape[2], test_xt.shape[1]), dtype=float)
    FOLD_NUM = train_x.shape[0]
    TRIAL_NUM = train_x.shape[1]
    for i in range(FOLD_NUM):
        for j in range(TRIAL_NUM):
            train_x[i] = train_xt[i].transpose()
        for j in range(TRIAL_NUM//4):
            test_x[i] = test_xt[i].transpose()
    # reshape data y -->
    #   from (# of fold, # of frequency bands, # of classes, # of trials)
    #   to   (# of fold, # of trials, # of frequency bands, # of classes)
    train_yt = data['train_y']
    test_yt = data['test_y']
    train_y = np.zeros((train_yt.shape[0], train_yt.shape[2], train_yt.shape[1]))
    test_y = np.zeros((test_yt.shape[0], test_yt.shape[2], test_yt.shape[1]))
    for i in range(FOLD_NUM):
        train_y[i] = train_yt[i].transpose()
        test_y[i] = test_yt[i].transpose()
    return train_x, test_x, train_y, test_y


def make_model(input_size):
    model = Sequential()
    model.add(Dense(20, input_dim=input_size, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def test():
    path = 'D:\\BCI-comparative-analysis\\RA\\5c_f\\csp\\'
    file_list = read_dir(path)
    for fname in file_list:
        mat_file = read_file(path+fname)
        csp = mat_file['csp']
        train_x, test_x, train_y, test_y = read_data(csp)
        fold = range(train_x.shape[0])

        for i in fold:
            # make CNN model
            model = make_model((train_x.shape[2]))
            # train model
            model.fit(x=train_x[i], y=train_y[i], batch_size=32, epochs=200, verbose=1, validation_data=(test_x[i], test_y[i]))
            model.save('D:\\BCI-comparative-analysis\\model\\'+fname[:-4]+str(i+1)+'.h5')


if __name__ == '__main__':
    test()
