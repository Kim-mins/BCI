'''
gdf2mat.py:
    convert .gdf file into .mat file
'''
import mne
import io
import os
import numpy as np
import scipy.io as sio


# the # of classes
CLASS_NUM = 4
# the # of EEG channels
CHANNEL_NUM = 22
# sampling frequency
S_FREQ = 250.0
# sampling interval
TIME_INTERVAL = [0.5, 2.5]
TIME_OFFSET = TIME_INTERVAL[1] - TIME_INTERVAL[0]
# # of trials
TRIAL_NUM = 288


# read_dir(path):
#   reads directory to get file(data) names
def read_dir(path):
    # file_list: list of file names
    file_list = os.listdir(path)
    return file_list


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


# num2one_hot(num):
#   convert number to one-hot encoding, num is 'zero based'(class 1 --> 0, class 2 --> 1, ...)
def num2one_hot(num):
    one_hot = []
    for i in range(CLASS_NUM):
        if i == num:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return np.asarray(one_hot)


# proc_selectClasses(data, label):
#   group data by class
def proc_selectClasses(data, label):
    # placeholder for data of each class
    data_group = [[] for i in range(CLASS_NUM)]
    # label for each class
    class_group = [[] for i in range(CLASS_NUM)]
    for i in range(TRIAL_NUM):
        # class
        c = label[i]
        # group data by class
        data_group[c].append(data[i])
        # label also.
        one_hot = num2one_hot(c)
        class_group[c].append(one_hot)
    # grouped data
    ged_data = []
    # grouped label
    ged_label = []
    # append ascending order(class)
    for i in range(CLASS_NUM):
        for j in range(TRIAL_NUM//4):
            ged_data.append(data_group[i][j])
            ged_label.append(class_group[i][j])
    return ged_data, ged_label


# get_dl(*):
#   get data & label:
def get_dl(signal, events):
    # start time for each trials(for time segmentation)
    start_times = []
    # label
    label = []
    for event in events:
        # offset is event code
        offset = event[2] - event[1]
        if(offset == 769 or offset == 770 or offset == 771 or offset == 772):
            # 768: nothing
            # 769: event code of left hand --> class 1
            # 770: event code of right hand --> class 2
            # 771: event code of foot --> class 3
            # 772: event code of tongue --> class 4
            # 0.5 for starting time --> [0.5 2.5] is time interval
            # event[0] == time in (sec * sampling frequency) scale
            time = event[0]
            # multiply sampling frequency to sampling starting time(for scaling)
            start_times.append(time + int(TIME_INTERVAL[0]*S_FREQ))
            # class
            c = offset - 769
            # save label for each data
            label.append(c)
    # filter data for each time intervals
    data = []
    for start in start_times:
        # end time for each trials
        end = start + int(S_FREQ*TIME_OFFSET)
        # temporary data((channel&trial)wise)
        a = signal[:, start:end]
        data.append(a)
    data, label = proc_selectClasses(data, label)
    return np.asarray(data), np.asarray(label)


# get_dlc():
#   get data, labels, channel labels(for train)
def get_dlc(raw):
    # get sampling frequency
    sampling_frequency = raw.info["sfreq"]
    assert sampling_frequency == S_FREQ
    # get events list
    events = mne.find_events(raw, initial_event = True)
    # get channel names
    clab = raw.ch_names
    # need only EEG channels(#: 22)
    clab = np.asarray(clab[:CHANNEL_NUM])
    # load signal data for all the channels
    signal = raw._data
    # need only 22 channels' data(EEG channels)
    signal = signal[:CHANNEL_NUM]
    # bandpass filtering
    filtered_signal = [[] for i in range(CHANNEL_NUM)]
    for channel in range(CHANNEL_NUM):
        filtered_signal[channel] = bandpass_filter(signal[channel], 4, 40, fs=250.0, order=3)
    filtered_signal = np.asarray(filtered_signal)
    # get data & label
    data, label = get_dl(filtered_signal, events)
    # data.shape -> (# of trials (== 288), # of channels (== 22), # of signals(2(seconds) * 250(sampling_frequency) == 500)
    assert data.shape == (TRIAL_NUM, CHANNEL_NUM, S_FREQ*TIME_OFFSET)
    # label.shape -> (# of trials (== 288), # of CLASSES)
    assert label.shape == (TRIAL_NUM, CLASS_NUM)
    # clab.shape -> (# of channels (== 22),)
    assert clab.shape == (CHANNEL_NUM, )
    # Since matlab is column major, we should transpose data to make trial dimension as last dimension.
    return data.T, label.T, clab


# load_data(path):
#   load gdf file data
def load_data(path):
    files = read_dir(path)[:-1]
    # data for all the subjects
    sub_data = []
    # label for all the subjects
    sub_label = []
    # every subject has the same channel labels
    sub_clab = []
    # for each subjects...
    for name in files:
        print(name)
        # read raw gdf file
        raw = mne.io.read_raw_edf(path+name, preload=True)
        # get data, label, clab for one subject
        data, label, clab = get_dlc(raw)
        # data.shape -> (# of trials (== 288), # of channels (== 22), # of signals(2(seconds) * 250(sampling_frequency) == 500)
        assert data.shape == (S_FREQ*TIME_OFFSET, CHANNEL_NUM, TRIAL_NUM)
        # label.shape -> (# of trials (== 288),)
        assert label.shape == (CLASS_NUM, TRIAL_NUM)
        # clab.shape -> (# of channels (== 22),)
        assert clab.shape == (CHANNEL_NUM, )
        sub_data.append(data)
        sub_label.append(label)
        sub_clab = clab
    return np.asarray(sub_data), np.asarray(sub_label), np.asarray(sub_clab)


# save_data(path):
#   saves data as .mat file
def save_data(path, data, label, clab):
    mat_dict = {}
    mat_dict['data'] = data
    mat_dict['label'] = label
    mat_dict['clab'] = clab
    sio.savemat(path+'data', mat_dict)


if __name__ == '__main__':
    path='D:\\BCI-comparative-analysis\\BCICIV_2a_gdf\\train\\'
    data, label, clab = load_data(path)
    save_data(path+'mat\\', data, label, clab)
