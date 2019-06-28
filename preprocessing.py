from scipy.signal import butter, lfilter
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import glob
import wfdb

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    files = glob.glob('data/*.dat')
    i=0
    for record in files:
        if i>0:
            break
        print(record)
        record = record[:-4]
        signals, fields = wfdb.rdsamp(record, channels=[0])
        i += 1
    x = signals
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 360.0
    lowcut = 0.5
    highcut = 100.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    #T = 180
    #nsamples = T * fs
    #t = np.linspace(0, T, nsamples, endpoint=False)
    #a = 0.02
    t = np.arange(x.shape[0])
    f0 = 600.0

    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    #plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


#run()

def gain_reduction(signal):
    gain_val = 200
    return signal/gain_val

def rescaling(signal):
    return minmax_scale(signal, feature_range=(-1, 1))

def rescaling_paper(signal):
    mi = np.amin(signal)
    ma = np.amax(signal)
    for i in range(0, len(signal)):
        signal[i] = (2 * ( (signal[i] - mi) / (ma - mi) ) ) - 1
    return signal

def const_comp_reduction(signal):
    mean = np.mean(signal)
    #return signal/mean
    return signal - mean

def standardize(signal):
    mean  = np.mean(signal)
    std = np.std(signal)
    return (signal - mean)/std

def preprocess(signals, type):
    for i in range(0, len(signals)):
        signals[i] = gain_reduction(signal=signals[i])
        #signals[i] = const_comp_reduction(signal=signals[i])
        if type == 2:
            signals[i] = rescaling_paper(signal = signals[i])
            #signals[i] = rescaling(signal= signals[i])
        elif type == 3:
            signals[i] = standardize(signal = signals[i])
    print('Pre Processing Done')
    return signals


    print('hj')
'''import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({
        'data' : y_sine} ,index=t_sine)
    return result

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

fps = 30
sine_fq = 10 #Hz
duration = 10 #seconds
sine_5Hz = sine_generator(fps,sine_fq,duration)
sine_fq = 1 #Hz
duration = 10 #seconds
sine_1Hz = sine_generator(fps,sine_fq,duration)

sine = sine_5Hz + sine_1Hz

filtered_sine = butter_highpass_filter(sine.data,10,fps)

plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(range(len(sine)),sine)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('filtered signal')
plt.show()'''