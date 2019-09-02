import glob
import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

count = {'N': 0,
             'A': 0,
             '(AFL': 0,
             '(AFIB': 0,
             '(SVTA': 0,
             '(PREX': 0,
             'V': 0,
             '(B': 0,
             '(T': 0,
             '(VT': 0,
             '(IVR': 0,
             '(VFL': 0,
             'F': 0,
             'L': 0,
             'R': 0,
             '(BII': 0,
             '/': 0}

knowledge = {'N': 'Normal_sinus_rhythm',
             'A': 'Atrial_premature_beat',
             '(AFL': 'Atrial_flutter',
             '(AFIB': 'Atrial_fibrillation',
             '(SVTA': 'Supraventricular_tachyarrhythmia',
             '(PREX': 'Pre-excitation_(WPW)',
             'V': 'Premature_ventricular_contraction',
             '(B': 'Ventricular_bigeminy',
             '(T': 'Ventricular_trigeminy',
             '(VT': 'Ventricular_tachycardia',
             '(IVR': 'Idioventricular_rhythm',
             '(VFL': 'Ventricular_flutter',
             'F': 'FusionIdioventricular_of_ventricular_and_normal_beat',
             'L': 'Left_bundle_branch_block_beat',
             'R': 'Right_bundle_branch_block_beat',
             '(BII': 'Second-degree_heart_block',
             '/': 'Pacemaker_rhythm'}

beat_data = []
beat_labels = []
beats_data = {}
X = []
y = []
def add_to_data(signals, i, label, beats):
    if i != 0 and i != len(beats) - 1:
        signal_idx = beats[i]
        if signal_idx > 100 and signal_idx < len(signals) - 200:
            data = signals[signal_idx - 100: signal_idx + 200]
            done = signal_idx + 200 - 1
            '''directory = 'samples/' + knowledge[label]
            if not os.path.isdir(directory):
                os.makedirs(directory)
            count = len(os.listdir(directory))
            filename = directory + '/' + str(count + 1) + '.jpg'
            y_plt = np.arange(len(data))
            x_plt = data
            fig = plt.figure(frameon=False)
            plt.plot(np.array([i[0] for i in data]))
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            fig.savefig(filename)
            plt.close(fig=fig)

            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)'''

            if label in beats_data:
                beats_data[label].append(data.reshape(1,-1))
            else:
                beats_data[label] = [data.reshape(1,-1)]
            return True, done
    return False, None

def read_data():
    # read all files in data dir
    files = glob.glob('data/*.dat')

    for record in files:
        print(record)
        record = record[:-4]
        signals, fields = wfdb.rdsamp(record, channels = [0])
        annotation = wfdb.rdann(record, 'atr')

        beats = list(annotation.sample)
        done = 0
        for i in range(0, len(annotation.symbol)):
            if annotation.symbol[i] in count:
                count[annotation.symbol[i]] += 1
                result, d = add_to_data(signals, i, annotation.symbol[i], beats)
                if result:
                    done = d
            elif annotation.symbol[i] == '+':
                if annotation.aux_note[i].strip('\x00') in count:
                    if annotation.aux_note[i].strip('\x00') == '(VFL' or annotation.aux_note[i].strip('\x00') == '(BII':
                        print('Found')
                    count[annotation.aux_note[i].strip('\x00')] += 1
                    result, d = add_to_data(signals, i, annotation.aux_note[i].strip('\x00'), beats)
                    if result:
                        done = d
            #else:
                #print("Symbol not present: "+annotation.symbol[i])
        #print('yolo')
    print('done')
    sum = 0
    threshold = 5000
    new_data = {}
    for key, pair in beats_data.items():
        print(key+' -count is: '+str(len(pair)))
        if len(pair) >= threshold:
            random.shuffle(pair)
            p = np.array(pair[:5000])
            nsamples, nx, ny = p.shape
            new_data[key] = p.reshape((nsamples, nx * ny))
        else:
            a = pair
            b = []
            while len(b) < threshold:
                b.extend(a)
            random.shuffle(b)
            p = np.array(b[:5000])
            nsamples, nx, ny = p.shape
            new_data[key] = p.reshape((nsamples, nx * ny))
        sum += len(pair)
    print('Total - ' + str(sum))
    X = []
    y = []
    for key, pair in new_data.items():
        X.extend(pair)
        for i in range(0, len(pair)):
            y.append(key)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    a,b = read_data()