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
from matplotlib import pyplot as plt
np.random.seed(7)


# Tell if print train/test log or not
log = True
# Tell if print shape of the data
shape = False
# Tell if train is needed
TRAIN = True
# Tell if test is needed
TEST = False
# Tell if score is needed
SCORE = False
# # of subject
SUBJECT_NUM = 9
# # of fold(== K)
K = 4
# # of trials 
TRIAL_NUM = 288


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
    # reshape data x --> 
    #   from (# of folds, # of frequency bands, # of pairwise matrix, # of trials)
    #   to   (# of folds, # of trials, # of frequency bands, # of pairwise matrix)
    # *_xt is temporary value for train & test
    a = data['train_x']
    b = data['test_x']
    c = data['train_y']
    d = data['test_y']
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    for sub in range(SUBJECT_NUM):
        # tmp1: train_x temporary value
        tmp1 = []
        # tmp2: test_x temporary value
        tmp2 = []
        # tmp3: train_y temporary value
        tmp3 = []
        # tmp4: test_y temporary value
        tmp4 = []
        for fold in range(K):
            # *_t: temporary values
            train_t = a[sub][fold].T
            test_t = b[sub][fold].T
            train_t_y = c[sub][fold].T
            test_t_y = d[sub][fold].T
            train_xt = np.zeros((train_t.shape[0], train_t.shape[2], train_t.shape[1]), dtype=float)
            test_xt = np.zeros((test_t.shape[0], test_t.shape[2], test_t.shape[1]), dtype=float)
            for i in range(TRIAL_NUM*(K-1)//K):
                train_xt[i] = train_t[i].T
            for i in range(TRIAL_NUM//K):
                test_xt[i] = test_t[i].T
            train_xt = np.reshape(train_xt, (train_xt.shape[0], train_xt.shape[1], train_xt.shape[2], 1))
            test_xt = np.reshape(test_xt, (test_xt.shape[0], test_xt.shape[1], test_xt.shape[2], 1))
            tmp1.append(train_xt)
            tmp2.append(test_xt)
            tmp3.append(train_t_y)
            tmp4.append(test_t_y)
        train_x.append(tmp1)
        test_x.append(tmp2)
        train_y.append(tmp3)
        test_y.append(tmp4)
    train_x = np.asarray(train_x)
    test_x = np.asarray(test_x)
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    if zero_or_inf:
        res = any_zero_or_inf(train_x)
        print(res)
    # print the shape of data
    if shape:
        print_data_shape(train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y


# make_CNN(*):
#   make Convolution Neural Network model
def make_CNN(input_dim):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(1, 5), input_shape=(input_dim[0], input_dim[1], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(6, kernel_size=(1, 5), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(5, kernel_size=(3, 3), activation='tanh'))
    model.add(Flatten())
#    model.add(Dense(64, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.003), metrics=['accuracy'])
#    model.summary()
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
    for i in range(TRIAL_NUM*(K-1)//K):
        train_x[i] = train_xt[i].transpose()
    for i in range(TRIAL_NUM//K):
        test_x[i] = test_xt[i].transpose()
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2]))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2]))
    # reshape data y -->
    #   from (# of fold, # of frequency bands, # of classes, # of trials)
    #   to   (# of fold, # of trials, # of frequency bands, # of classes)
    train_y = data['train_y'].transpose()
    test_y = data['test_y'].transpose()
    # flatten data
    train_x = flatten_x(train_x)
    test_x = flatten_x(test_x)
    train_y = one_hot2num(train_y)
    test_y = one_hot2num(test_y)
    # print the shape of data
    if shape:
        print_data_shape(train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y


# train_CNN(*):
#   train CNN and save that model
def train_CNN(model_path, score_path, fname, train_x, test_x, train_y, test_y):
#    from keras.models import model_from_json
    acc = []
    val_acc = []
    for sub in range(SUBJECT_NUM):
        acc_t = []
        val_acc_t = []
        for fold in range(K):
            # make CNN model
            model = make_CNN(input_dim=(train_x[sub][fold].shape[1], train_x[sub][fold].shape[2]))
            # train model
            hist = model.fit(x=train_x[sub][fold], y=train_y[sub][fold], batch_size=38, epochs=30, verbose=1 if log else 0, validation_data=(test_x[sub][fold], test_y[sub][fold]))
            acc_t.append(hist.history['acc'][-1])
            val_acc_t.append(hist.history['val_acc'][-1])
        acc.append(acc_t)
        val_acc.append(val_acc_t)
    np.save(score_path+fname[:-4]+'_acc', acc)
    np.save(score_path+fname[:-4]+'_val_acc', val_acc)
    model.save(model_path+fname[:-4]+'.h5')
#            model_json = model.to_json()
#            with open(model_path+fname+'_'+str(sub)+'_'+str(fold)+'.json', "w") as json_file:
#                json_file.write(model_json)
#            h5_path=weight_path+fname+".h5"
#            model.save_weights(h5_path,overwrite="True")


# test_CNN(*):
#   test all the trained CNN-model with test data, and save loss & accuracy
def test_CNN(model_path, score_path, test_x, test_y):
    # model name list
    model_list = read_dir(model_path)
    # loss, accuracy list
    loss = []
    acc = []
    for sub in range(SUBJECT_NUM):
        loss_t = []
        acc_t = []
        for fold in range(K):
            # load model
            model = load_model(model_path+model_list[sub*K+fold])
            # evaluate model with test data(test_x, test_y)
            loss_t1, acc_t1 = model.evaluate(test_x[sub][fold], test_y[sub][fold])
            loss_t.append(loss_t1)
            acc_t.append(acc_t1)
        loss.append(loss_t)
        acc.append(acc_t)
    np.save(score_path+'acc', acc)
    np.save(score_path+'loss', loss)


# score_CNN(*):
#   get evaluated score(loss, accuracy) of CNN
def score_CNN(score_path):
    loss = np.load(score_path+'loss.npy')
    acc = np.load(score_path+'acc.npy')
    paper_acc = [79.16, 52.08, 83.33, 62.15, 54.51, 39.24, 83.33, 82.64, 66.67]
    mean = stat(acc)
    for sub in range(SUBJECT_NUM):
        diff = mean[sub]*100 - paper_acc[sub]
        print('sub{:d}: mean={:.2f}'.format(sub+1, mean[sub]*100))
        print('diff: {:.2f}'.format(diff))
    return loss, acc


def CNN():
    print('CNN')
    # name your path
    path = '../acsp_comp/'
    # path to save model
    model_path = '../model/'
    # weight path
    weight_path = 'E:/Robot arm/comp/weight/'
    # path to save, load score
    score_path = '../score/'
    # read directory
    file_list = read_dir(path)
    file_list = file_list[144:]
    for fname in file_list:
        # read .mat file(data)
        mat_file = read_file(path+fname)
        # load struct, 'features' --> using matlab struct type
        features = mat_file['features']
        # read train, test data
        train_x, test_x, train_y, test_y = read_data_CNN(features)
        print(train_x.shape)
        print(test_x.shape)
        # train model
        if TRAIN:
            train_CNN(model_path, score_path, fname, train_x, test_x, train_y, test_y)
        # evaluate model
        if TEST:
            test_CNN(model_path, score_path, test_x, test_y)
        if SCORE:
            score_CNN(score_path)

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
    machine_epsilon = 7./3 - 4./3 - 1
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    for m in range(x.shape[4]):
                        if np.absolute(x[i][j][k][l][m]) < machine_epsilon:
                            count_z = count_z + 1
    # test if any infs are in 'x'
    count_inf = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    for m in range(x.shape[4]):
                        if np.absolute(x[i][j][k][l][m]) < np.inf:
                            count_inf = count_inf + 1
    if count_z > 0:
        return 'zero'
    if count_inf < x.shape[0]*x.shape[1]*x.shape[2]:
        return 'inf'
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


# plottor(*):
#   plot data
def plottor(loss, acc):
    plt.plot(loss, label='loss')
    plt.plot(acc, label='acc')


# stat(*);
#   get statistical values based on data
def stat(data):
    mean = np.mean(data, axis=1)
    # ...
    return mean


if __name__ == '__main__':
    CNN()
#    LSVM()
#    KSVM()
#    GB()
#    LDA()


# m = 2일 때 매우 잘 됨