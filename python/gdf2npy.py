'''
gdf2npy.py -- convert .gdf file into .npy
'''
from matplotlib import pyplot as plt
import numpy as np

def load_data(path):
    import mne
    raw = mne.io.read_raw_edf(path, preload = True)
    sampling_frequency = raw.info["sfreq"]
    events = mne.find_events(raw, initial_event = True)
    label = []
    start_times = []
    for line in events:
        if(line[2] - line[1] == 769 or line[2] - line[1] == 770):
            # 769: event code of left hand
            # 770: event code of right hand
            start_times.append(float(line[0])/sampling_frequency + 0.5)
            label.append([1, 0] if line[2] - line[1] == 769 else [0, 1])
    C3 = np.asarray([])
    Cz = np.asarray([])
    C4 = np.asarray([])
    for time in start_times:
        # time != times
        # times is for plotting of raw data
        start = int(sampling_frequency * time)
        end = int(sampling_frequency * (time + 2.0))
        if end - start > 500:
            start += 1
        temp, _ = raw[:3, start:end] # _ is times
        C3 = np.asarray(C3.tolist() + np.asarray([temp[0]]).tolist())
        Cz = np.asarray(Cz.tolist() + np.asarray([temp[1]]).tolist())
        C4 = np.asarray(C4.tolist() + np.asarray([temp[2]]).tolist())
    data = np.asarray(np.asarray([C3]).tolist() + np.asarray([Cz]).tolist() + np.asarray([C4]).tolist())
    label = np.asarray(label)
    # data.shape -> (# of channels (== 3), # of trials (== 120 or 160), # of signals(2(seconds) * 250(sampling_frequency) = 500)
    assert data.shape == (3, len(label), 500)
    return data, label

# Short Time Fourier Transform
def short_time_fourier_transform(data):
    from scipy import signal
    sampling_frequency = 250.0
    window_size = 64    # 0.24sec for each windows
    time_lapse = 14     # update period = 14
    overlap_size = window_size - time_lapse
    frequency, time, spec = signal.stft(x = data, fs = sampling_frequency, nperseg = window_size, nfft = 512, noverlap = overlap_size)
    # x:  input signal
    # fs: sampling frequency
    # nperseg: window size
    # noverlap: number of overlap segment
    # nfft: I don't know why nfft should be 512
    
    assert spec.shape == (3, data.shape[1], 257, 37)
    return spec

def resize_image_to_93_37(spec):
    # spec: 'STFTed' image
    # spec[0]: channel C3, spec[1]: channel Cz, spec[2]: channel C4
    import cv2
    x = 37
    y = 15
    result = np.asarray([])
    for i in range(len(spec[0])):
        channel = np.asarray([])
        for j in range(3):
            # filter mu-band(6 ~ 13Hz) & beta-band(17 ~ 30Hz)(approx)
            mu_band = np.abs(spec[j][i][12:28])
            beta_band = np.abs(spec[j][i][35:63])
            # cubic_interpolation(resize beta-band to the size of 15 x 37)
            new_beta_band = cv2.resize(beta_band, dsize = (x, y), interpolation = cv2.INTER_CUBIC)
            # concat two bands(6 ~ 13 / 17 ~ 30) --> image of one channel(size = 31 x 37)
            temp = np.asarray(mu_band.tolist() + new_beta_band.tolist())
            channel = np.asarray(channel.tolist() + temp.tolist())
        result = np.asarray(result.tolist() + np.asarray([channel]).tolist())
    
    assert result.shape == (spec.shape[1], 93, 37)
    return result
    
# plotting raw signal data(C3, Cz, C4)
def plot_raw_signal_data(time_with_data):
    from matplotlib import pyplot as plt
    # time_with_data[0]: times
    # time_with_data[1]: data
    plt.plot(time_with_data[0], time_with_data[1].T)
    plt.title("Sample channels")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.show()

def plot_spectrogram(spec, x, y):
    from matplotlib import pyplot as plt
    plt.pcolormesh(x, y, spec) # 14:62 == 7~30Hz    12:28 == 6 ~ 13Hz    35:63 == 17 ~ 30Hz
    plt.show()
'''
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
'''
def get_file_name(i, j):
    name = "B0" + str(i) + "0" + str(j) + "T"
    return name

def save_as_npy(name, image, label):
    np.save("../npy/" + name, image)
    np.save("../npy/" + name + "_y", label)

if __name__ == "__main__":
    for i in range(1, 10):
        for j in range(1, 4):
            path = "../data/"
            name = get_file_name(i, j)
            path += name
            path += ".gdf"
            print("************************************************")
            print("|              processing", name, "              |")
            print("************************************************")
            # load raw gdf data
            data, label = load_data(path)
            # short time fourier transform
            spec = short_time_fourier_transform(data)
            # resize image to the size of 93 x 37
            image = resize_image_to_93_37(spec)
            # save as .npy
            save_as_npy(name, image, label)


'''
참고자료
https://cbrnr.github.io/2017/10/23/loading-eeg-data/
https://mne-tools.github.io/0.14/auto_tutorials/plot_object_raw.html
http://www.incosys.co.kr/index.php/book/1-basic-tech-vibration/2015-01-21-07-36-51-6/2015-01-21-07-36-51-6-1/2015-01-21-07-36-51-6-1-5
https://blog.naver.com/bdh0727/221266874029
'''