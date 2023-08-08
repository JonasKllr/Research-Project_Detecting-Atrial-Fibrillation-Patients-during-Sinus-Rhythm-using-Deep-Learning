import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import wfdb
import wfdb.processing

from scipy.io import loadmat


def get_patient_number(filename):
    
    regex = re.compile(r'\d+')
    patient_number = [int(x) for x in regex.findall(filename)][0]
    
    return patient_number


    



# load data from PAF prediction challenge
def load_dataset_PAF(DIRECTORY):

    RECORD_DURATION_SECONDS = 1800
    FREQUENCY_HERTZ = 128

    signals = np.empty((0, 10 * FREQUENCY_HERTZ, 2))
    labels = np.empty(0)

    for filename in sorted(os.listdir(DIRECTORY)):
        
        # only do data integration once per record
        if filename.endswith('.dat'):
            
            # for files named "pXX" label = 1 (AF group)
            if (os.path.basename(filename)[0] == 'p'):
                filename_without_ext = os.path.splitext(filename)[0]
                file_directory = DIRECTORY + os.sep + filename_without_ext

                signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=RECORD_DURATION_SECONDS*FREQUENCY_HERTZ)[0]
                
                # fix ouliers in the last sample
                LAST_SAMPLE = 230399
                signals_temp[LAST_SAMPLE] = signals_temp[LAST_SAMPLE-1]

                # split records into 10 sec 
                signals_temp = np.split(signals_temp, indices_or_sections=180)
                
                # fill resulting array
                signals = np.append(signals, signals_temp, axis=0)

                lables_temp = np.full(180, 1)   # TODO make length dependent on length of signals_temp
                labels = np.append(labels, lables_temp)


            # for files named "pXX" label = 0 (non AF group)
            elif (os.path.basename(filename)[0] == 'n'):
                filename_without_ext = os.path.splitext(filename)[0]
                file_directory = DIRECTORY + os.sep + filename_without_ext

                signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=RECORD_DURATION_SECONDS*FREQUENCY_HERTZ)[0]
                
                # fix ouliers in the last sample
                LAST_SAMPLE = 230399
                signals_temp[LAST_SAMPLE] = signals_temp[LAST_SAMPLE-1]

                # split records into 10 sec 
                signals_temp = np.split(signals_temp, indices_or_sections=180)
                
                # fill resulting array
                signals = np.append(signals, signals_temp, axis=0)

                lables_temp = np.full(180, 0)   # TODO make length dependent on length of signals_temp
                labels = np.append(labels, lables_temp)


            else:
                pass

    return signals, labels



# Get comments from header files under 'Dx'. Labels of the recording.
def get_header_comments_Dx_CinC(RECORD_DIR):
    
    header = wfdb.rdheader(RECORD_DIR)
    header_comments = header.comments
    
    # get entry under 'Dx' and convert into list with one label per element.
    comment_list = list()
    for comment in header_comments:
        if comment.startswith('Dx'):
            try:
                entries = comment.split(': ')[1].split(',')
                for entry in entries:
                    comment_list.append(entry.strip())
            except:
                pass
    
    return comment_list


# Get comments from header files under 'Age'. Used for labels in training
def get_header_comments_Age_CinC(RECORD_DIR):
    
    header = wfdb.rdheader(RECORD_DIR)
    header_comments = header.comments
    
    # get entry under 'Age' and convert to int.
    for comment in header_comments:
        if comment.startswith('Age'):
            try:
                age = comment.split(': ')[1]
                age = int(age)
                
            except:
                age = np.nan
                #pass
    
    return age



# https://github.com/physionetchallenges/python-classifier-2021/blob/main/helper_code.py#L65
def load_dataset_CinC(DIRECTORY):

    print('Loading CinC data set ...')

    FREQUENCY_HERTZ_TARGET = 128
    FREQUENCY_HERTZ_ORIGINAL = 500

    signals = np.empty((0, 10 * FREQUENCY_HERTZ_TARGET, 2))
    labels = np.empty(0)

    # for reproducibility
    random.seed(42)

    for filename in sorted(os.listdir(DIRECTORY)):
        
        # only do data integration once per record
        if filename.endswith('.mat'):

            print(filename)
            
            # read in .mat file
            file_dir_mat = DIRECTORY + os.sep + filename
            signals_temp = loadmat(file_dir_mat)['val']

            # pick 2 leads out of 12 randomly
            random_list = random.sample(range(11), 2)
            signals_temp = signals_temp[random_list]
            signals_temp = np.asarray(signals_temp)
            
            # change axis to realize shape (5000, 2)
            signals_temp = np.transpose(signals_temp)

            # resample to 128 Hz
            signals_temp_resampled = np.zeros((1, 1280,2))
            signals_temp_resampled[0,:,0] = wfdb.processing.resample_sig(signals_temp[:,0], FREQUENCY_HERTZ_ORIGINAL, FREQUENCY_HERTZ_TARGET)[0]
            signals_temp_resampled[0,:,1] = wfdb.processing.resample_sig(signals_temp[:,1], FREQUENCY_HERTZ_ORIGINAL, FREQUENCY_HERTZ_TARGET)[0]

            signals = np.append(signals, signals_temp_resampled, axis=0)

            # get age of the patient
            filename_without_ext = os.path.splitext(filename)[0]
            file_dir_hea = DIRECTORY + os.sep + filename_without_ext
            age = get_header_comments_Age_CinC(file_dir_hea)

            if age in range(0,10):
                labels = np.append(labels, 0)
            elif age in range(10,20):
                labels = np.append(labels, 1)
            elif age in range(20,30):
                labels = np.append(labels, 2)
            elif age in range(30,40):
                labels = np.append(labels, 3)
            elif age in range(40,50):
                labels = np.append(labels, 4)
            elif age in range(50,60):
                labels = np.append(labels, 5)
            elif age in range(60,70):
                labels = np.append(labels, 6)
            elif age in range(70,80):
                labels = np.append(labels, 7)
            elif age in range(80,90):
                labels = np.append(labels, 8)
            elif age in range(90,100):
                labels = np.append(labels, 9)
            else:
                pass

        else:
            pass
    
    return signals, labels








# TESTS
if __name__ == '__main__':

    filename = 'n11.dat'
    get_patient_number(filename)



    
    #DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/test'
    DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_and_age-not-300_only'


    signals, labels = load_dataset_CinC(DIRECTORY)
    print(signals)
    print(np.shape(signals))

    print(labels)
    print(type(labels))
    print(np.shape(labels))

    SAVE_DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'
    #np.save(SAVE_DIR + 'CinC.npy', signals, allow_pickle=False)
    #np.save(SAVE_DIR + 'CinC_age.npy', labels, allow_pickle=False)






    # set record parameters
    RECORD_DURATION_SECONDS = 10
    FREQUENCY_HERTZ = 128.0


    # from here only for plotting the signal
    nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
    t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

    #plt.figure(2)
    fig, axs = plt.subplots(2)

    # plot normalized signals
    z = signals[0, :, 0]
    axs[0].plot(t, z)
    axs[0].set_title('lead 0 normalized', loc='left')
    axs[0].set_xlabel('seconds')
    
    w = signals[0, :, 1]
    axs[1].plot(t, w)
    axs[1].set_title('lead 1 normalized', loc='left')
    axs[1].set_xlabel('seconds')


    plt.tight_layout()
    plt.show()







    # Histogram
    #signals_flat, bins = np.histogram(signals, bins=500)
    #print(signals_flat.min())
    #print(signals_flat.max())

    #print(signals.min())
    #print(signals.max())
    
    #plt.bar(bins[:-1], signals_flat, width=np.diff(bins))
    #plt.ylim(0, 50)
    #plt.show()

