# ACSP.py -- model for 'ACSPed' data

# os for read directory
import os
# scipy.io for read .mat file
import scipy.io as sio
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
np.random.seed(7)


# Tell if class 'rest' is needed
# True: yes, False: no
rest = False
# Tell if print train/test log or not
log = True
# Tell if print shape of the data
shape = False

# K for cross valiation
K = 5


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


# read_data_CNN(features):
#   reads train, test data from features(matlab data) for CNN
def read_data_CNN(features, zero_or_inf=False):
    # to extract train, test set in matlab 'struct'
    data = features[0,0]
    res_train_x = []
    res_train_y = []
    res_test_x = []
    res_test_y = []
#    if zero_or_inf:
#        res = any_zero_or_inf(data['train_x'])
#        print(res)
#        res = any_zero_or_inf(data['test_x'])
#        print(res)
    for fold in range(K):
        # reshape data x --> 
        #   from (# of folds, # of frequency bands, # of pairwise matrix, # of trials)
        #   to   (# of folds, # of trials, # of frequency bands, # of pairwise matrix)
        # *_xt is temporary value for train & test
        train_xt = data['train_x'][fold].transpose()
        test_xt = data['test_x'][fold].transpose()
        train_x = np.zeros((train_xt.shape[0], train_xt.shape[2], train_xt.shape[1]))
        test_x = np.zeros((test_xt.shape[0], test_xt.shape[2], test_xt.shape[1]))
        TRIAL_NUM = train_x.shape[0]
        for i in range(TRIAL_NUM*(K-1)//K):
            train_x[i] = train_xt[i].transpose()
        for i in range(TRIAL_NUM//K):
            test_x[i] = test_xt[i].transpose()
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))
        # reshape data y -->
        #   from (# of fold, # of frequency bands, # of classes, # of trials)
        #   to   (# of fold, # of trials, # of frequency bands, # of classes)
        train_y = data['train_y'][fold].transpose()
        test_y = data['test_y'][fold].transpose()
        res_train_x.append(train_x)
        res_train_y.append(train_y)
        res_test_x.append(test_x)
        res_test_y.append(test_y)
    train_x = np.asarray(res_train_x)
    train_y = np.asarray(res_train_y)
    test_x = np.asarray(res_test_x)
    test_y = np.asarray(res_test_y)
    if not rest:
        train_x, test_x, train_y, test_y = no_rest(train_x, test_x, train_y, test_y)
    if zero_or_inf:
        res = any_zero_or_inf(train_x)
        res = any_zero_or_inf(test_x)
        print(res)
    # print the shape of data
    if shape:
        print_data_shape(train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y


# make_CNN(*):
#   make Convolution Neural Network model
def make_CNN(input_dim):
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(9, 9), input_shape=(input_dim[0], input_dim[1], 1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, kernel_size=(5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(5, kernel_size=(3, 3), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(5 if rest else 4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])
    model.summary()
    return model


# make_LSVM():
#   create Linear Support Vector Machine model
def make_LSVM():
    model = LinearSVC()
    return model


# make_KSVM():
#   create Kernel Support Vector Machine model
def make_KSVM():
    model = SVC(kernel='rbf')
    return model


# make_GB():
#   create Gradient Boosting model
def make_GB():
    model = GradientBoostingClassifier()
    return model


# make_LDA():
#   create (shrinkage) Linear Discriminant Analysis model
def make_LDA():
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return model


# read_data_LSVM(features):
#   reads train, test data from features(matlab data) for LSVM, KSVM, Gradient Boosting, LDA
def read_data_LSVM(features):
    # to extract train, test set in matlab 'struct'
    data = features[0,0]
    # reshape data x --> 
    #   from (# of folds, # of frequency bands, # of pairwise matrix, # of trials)
    #   to   (# of folds, # of trials, # of frequency bands, # of pairwise matrix)
    # *_xt is temporary value for train & test
    train_xt = data['train_x'].transpose()
    test_xt = data['test_x'].transpose()
    train_x = np.zeros((train_xt.shape[0], train_xt.shape[2], train_xt.shape[1]), dtype=float)
    test_x = np.zeros((test_xt.shape[0], test_xt.shape[2], test_xt.shape[1]), dtype=float)
    TRIAL_NUM = train_x.shape[0] + test_x.shape[0]
    for i in range(TRIAL_NUM):
        train_x[i] = train_xt[i].transpose()
    for i in range(TRIAL_NUM//4):
        test_x[i] = test_xt[i].transpose()
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2]))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2]))
    # reshape data y -->
    #   from (# of fold, # of frequency bands, # of classes, # of trials)
    #   to   (# of fold, # of trials, # of frequency bands, # of classes)
    train_y = data['train_y'].transpose()
    test_y = data['test_y'].transpose()
    # rest
    if not rest:
        train_x, test_x, train_y, test_y = no_rest(train_x, test_x, train_y, test_y)
    # flatten data
    train_x = flatten_x(train_x)
    test_x = flatten_x(test_x)
    train_y = one_hot2num(train_y)
    test_y = one_hot2num(test_y)
    # print the shape of data
    if shape:
        print_data_shape(train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y


# no_rest():
#   get data with no 'rest class' --> into 4-class
def no_rest(train_x, test_x, train_y, test_y):
    e = train_x.shape[1]//2
    f = e//(K-1)
    train_x = np.delete(train_x, np.s_[-e:], 1)
    train_y = np.delete(train_y, np.s_[-e:], 1)
    train_y = np.delete(train_y, np.s_[-1:], 2)
    test_x = np.delete(test_x, np.s_[-f:], 1)
    test_y = np.delete(test_y, np.s_[-f:], 1)
    test_y = np.delete(test_y, np.s_[-1:], 2)
    return train_x, test_x, train_y, test_y


def CNN():
    print('CNN')
    # name your path
    path = 'D:\\BCI-comparative-analysis\\RA\\5c_f\\acsp\\'
    # read directory
    file_list = read_dir(path)[:-2]
    # for each file name...
    cnt = 0
    # train, test log
    train_log = []
    test_log = []
    for fname in file_list:
        # read .mat file(data)
        mat_file = read_file(path+fname)
        # load struct, 'features' --> using matlab struct type
        features = mat_file['features']
        # read train, test data
        train_x, test_x, train_y, test_y = read_data_CNN(features)
        for fold in range(train_x.shape[0]):
            # make CNN model
            model = make_CNN(input_dim=(train_x.shape[2], train_x.shape[3]))
            # train model
            hist = model.fit(x=train_x[fold], y=train_y[fold], batch_size=40, epochs=30, verbose=1 if log else 0, validation_data=(test_x[fold], test_y[fold]))
            model.save('D:\\BCI-comparative-analysis\\model\\'+fname[:-4]+str(cnt+1)+'.h5')
            train_log.append(hist.history['acc'][-1])
            test_log.append(hist.history['val_acc'][-1])
            cnt = cnt + 1
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_CNN_train', train_log)
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_CNN_test', test_log)
#    predictor(path, test_x, test_y)


def LSVM():
    print('LSVM')
    # name your path
    path = 'D:\\BCI-comparative-analysis\\RA\\5c_f\\acsp\\'
    # read directory
    file_list = read_dir(path)
    # train, test log
    train_log = []
    test_log = []
    # for each file name...
    for fname in file_list:
        # read .mat file(data)
        mat_file = read_file(path+fname)
        # load struct, 'features' --> using matlab struct type
        features = mat_file['features']
        # read train, test data
        train_x, test_x, train_y, test_y = read_data_LSVM(features)
        # make CNN model
        model = make_LSVM()
        # train model
        model.fit(X=train_x, y=train_y)
        # print result
        if log:
            print_res(model, train_x, test_x, train_y, test_y)
        train_log.append(model.score(train_x, train_y))
        test_log.append(model.score(test_x, test_y))
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_LSVM_train', train_log)
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_LSVM_test', test_log)

def KSVM():
    print('KSVM')
    # name your path
    path = 'D:\\BCI-comparative-analysis\\RA\\5c_f\\acsp\\'
    # read directory
    file_list = read_dir(path)
    # train, test log
    train_log = []
    test_log = []
    # for each file name...
    for fname in file_list:
        # read .mat file(data)
        mat_file = read_file(path+fname)
        # load struct, 'features' --> using matlab struct type
        features = mat_file['features']
        # read train, test data
        train_x, test_x, train_y, test_y = read_data_LSVM(features)
        # make CNN model
        model = make_KSVM()
        # train model
        model.fit(X=train_x, y=train_y)
        # print result
        if log:
            print_res(model, train_x, test_x, train_y, test_y)
        train_log.append(model.score(train_x, train_y))
        test_log.append(model.score(test_x, test_y))
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_KSVM_train', train_log)
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_KSVM_test', test_log)


def GB():
    print('Gradient Boosting')
    # name your path
    path = 'D:\\BCI-comparative-analysis\\RA\\5c_f\\acsp\\'
    # read directory
    file_list = read_dir(path)
    # train, test log
    train_log = []
    test_log = []
    # for each file name...
    for fname in file_list:
        # read .mat file(data)
        mat_file = read_file(path+fname)
        # load struct, 'features' --> using matlab struct type
        features = mat_file['features']
        # read train, test data
        train_x, test_x, train_y, test_y = read_data_LSVM(features)
        # make CNN model
        model = make_GB()
        # train model
        model.fit(X=train_x, y=train_y)
        # print result
        if log:
            print_res(model, train_x, test_x, train_y, test_y)
        train_log.append(model.score(train_x, train_y))
        test_log.append(model.score(test_x, test_y))
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_GB_train', train_log)
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_GB_test', test_log)


def LDA():
    print('Shrinkage LDA')
    # name your path
    path = 'D:\\BCI-comparative-analysis\\RA\\5c_f\\acsp\\'
    # read directory
    file_list = read_dir(path)
    # train, test log
    train_log = []
    test_log = []
    # for each file name...
    for fname in file_list:
        # read .mat file(data)
        mat_file = read_file(path+fname)
        # load struct, 'features' --> using matlab struct type
        features = mat_file['features']
        # read train, test data
        train_x, test_x, train_y, test_y = read_data_LSVM(features)
        # make CNN model
        model = make_LDA()
        # train model
        model.fit(X=train_x, y=train_y)
        # print result
        if log:
            print_res(model, train_x, test_x, train_y, test_y)
        train_log.append(model.score(train_x, train_y))
        test_log.append(model.score(test_x, test_y))
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_LDA_train', train_log)
    np.save('D:\\BCI-comparative-analysis\\hist\\'+fname[:-4]+'_LDA_test', test_log)


'''
utils below
'''

def flatten_x(x):
    new_x = []
    for i in range(x.shape[0]):
        x_t = []
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_t.append(x[i][j][k])
        new_x.append(x_t)
    return np.array(new_x)


def one_hot2num(y):
    new_y = []
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j] == 1:
                new_y.append(j+1)
    return np.array(new_y)


# any_zeros(x):
#   tell if there's any zero or inf values
def any_zero_or_inf(x):
    # test if any zeros are in 'x'
    count_z = 0
    # test if any infs are in 'x'
    count_inf = 0
    machine_epsilon = 7./3 - 4./3 - 1
    for fold in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(x.shape[3]):
                    if np.absolute(x[fold][i][j][k]) < machine_epsilon:
                        print(str(fold+1)+' '+str(i+1)+' '+str(j+1)+' '+str(k+1))
                        count_z = count_z + 1
                    if np.absolute(x[fold][i][j][k]) < np.inf:
                        count_inf = count_inf + 1
    res = ''
    if count_z > 0:
        print('# of zeros: '+str(count_z))
        return res+'zero'
    if count_inf < x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]:
        print('# of infs: '+str(count_inf))
        return res+'inf'
    return 'gogo'


# print_data_shape(*):
#   print shape of train/test data
def print_data_shape(train_x, test_x, train_y, test_y):
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)


# print_res(*):
#   print result of learning
def print_res(model, train_x, test_x, train_y, test_y):
    print("train acc: {:.2f}".format(model.score(train_x, train_y)))
    print("test acc: {:.2f}".format(model.score(test_x, test_y)))


if __name__ == '__main__':
    CNN()
    LSVM()
    KSVM()
    GB()
    LDA()


# m = 2일 때 매우 잘 됨