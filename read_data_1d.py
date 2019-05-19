import glob
import wfdb

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
            if label in beats_data:
                beats_data[label].append(data)
            else:
                beats_data[label] = [data]
            X.append(data)
            y.append(label)
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
            else:
                print("Symbol not present: "+annotation.symbol[i])
        print('yolo')
    print('done')

    return X, y


'''
count = {'N': 0,
         'L': 0,
             'A': 0,
             'R': 0,
             'B': 0,
             'a': 0,
             'J': 0,
             'V': 0,
             'S': 0,
             'r': 0,
             'e': 0,
             'j': 0,
             'n': 0,
             'F': 0,
             'E': 0,
             '/': 0,
            'f': 0}


knowledge = {'N': 'Normal sinus rhythm',
             'A': 'Atrial premature beat',
             'AFL': 'Atrial flutter',
             'AFIB': 'Atrial fibrillation',
             'SVTA': 'Supraventricular tachyarrhythmia',
             'PREX': 'Pre-excitation (WPW)',
             'V': 'Premature ventricular contraction',
             'B': 'Ventricular bigeminy',
             'T': 'Ventricular trigeminy',
             'VT': 'Ventricular tachycardia',
             'IVR': ' rhythm',
             'VFL': 'Ventricular flutter',
             'F': 'FusionIdioventricular of ventricular and normal beat',
             'L': 'Left bundle branch block beat',
             'R': 'Right bundle branch block beat',
             'BII': 'Second-degree heart block',
             '/': 'Pacemaker rhythm'}'''