import os
import glob
import cv2
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# read all files in data dir
files = glob.glob('data\\*.dat')

# define the classes to extract the corresponding heartbeat
classes = ['N', 'L', 'R', 'E', '/', 'V', 'A', '!']

# map beat annotation code to class type for saving images
mapping = {'N': 'NOR', 'L': 'LBB', 'R': 'RBB', 'E': 'VEB', '/': 'PAB', 'V': 'PVC', 'A': 'APC', '!': 'VFB'}

# maintain the count of each sample encountered class-wise so images are not overwritten after each record
count = {'N': 0, 'L': 0, 'R': 0, 'E': 0, '/': 0, 'V': 0, 'A': 0, '!': 0}

for record in files:
    print(record)
    record = record[:-4]
    signals, fields = wfdb.rdsamp(record, channels = [0])  
    annotation = wfdb.rdann(record, 'atr')

    # dict to store the segments for each class
    segments = dict()
    
    # get indices for all beats
    beats = list(annotation.sample)

    for beat in classes:
        ids = np.in1d(annotation.symbol, beat)
        imp_beats = annotation.sample[ids]

        beat_samples = []

        for i in imp_beats:
            j = beats.index(i)
            if j != 0 and j != len(beats) - 1:
                
                # get the beats before and after this beat
                x = beats[j-1]

                if x + 20 > beats[j]:
                    l = 2
                    while (x + 20 > beats[j]) and (j-l >= 0):
                        x = beats[j-l]
                        l -= 1

                    if j-l < 0:
                        continue
                    
                y = beats[j+1]
                if y - 20 < beats[j]:
                    l = 2
                    while (y - 20 < beats[j]) and (j+l < len(beats) - 1):
                        y = beats[j+l]
                        l += 1

                    if j+l == len(beats) - 1:
                        continue

                # take subset at least 20 sec after and before above peaks x,y respectively
                start = x + 20
                end = y - 20

                # also, we need to centre the peak
                if abs(beats[j] - start) < abs(beats[j] - end):
                    end = beats[j] + abs(beats[j] - start)
                else:
                    start = beats[j] - abs(beats[j] - end)
                
                beat_samples.append(signals[start: end, 0])
        
        segments[beat] = beat_samples
    
    for key in segments.keys():
        if not segments[key]:
            continue
            
        val = segments[key]
        directory = 'samples\\' + mapping[key]
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        for i in val:
            fig = plt.figure(frameon=False)
            plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
        
            filename = directory + '\\' + str(count[key] + 1) + '.png'
            count[key] += 1
            fig.savefig(filename)
            plt.close(fig=fig)

            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)
    
    print('Completed record: ' + record)
        
print('done')


