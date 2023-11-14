import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb


def load_dataset_whole_recordings(DIRECTORY):

    RECORD_DURATION_SECONDS = 1800
    FREQUENCY_HERTZ = 128

    signals = np.empty((0, RECORD_DURATION_SECONDS * FREQUENCY_HERTZ, 2))
    labels = np.empty(0)
    patient_number_array= np.empty(0, dtype=np.int32)

    for filename in sorted(os.listdir(DIRECTORY)):
        
        # only do data integration once per record
        if filename.endswith('.dat'):
            
            # for files named "pXX" label = 1 (AF group)
            if (os.path.basename(filename)[0] == 'p'):
                filename_without_ext = os.path.splitext(filename)[0]
                file_directory = DIRECTORY + os.sep + filename_without_ext

                signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=RECORD_DURATION_SECONDS*FREQUENCY_HERTZ)[0]
                signals_temp = signals_temp[np.newaxis,:,:]
                # fix ouliers in the last sample
                #LAST_SAMPLE = 230399
                #signals_temp[LAST_SAMPLE] = signals_temp[LAST_SAMPLE-1]

                # split records into 10 sec 
                #signals_temp = np.split(signals_temp, indices_or_sections=180)
                
                # fill resulting array
                signals = np.append(signals, signals_temp, axis=0)

                # assign patient number to 10 sec segments
                #patient_number = get_patient_number(filename)
                #patient_number_array_temp = np.full(180, patient_number, dtype=np.int32)
                #patient_number_array = np.append(patient_number_array, patient_number_array_temp)

                lables_temp = np.full(1, 1)
                labels = np.append(labels, lables_temp)


            # for files named "pXX" label = 0 (non AF group)
            elif (os.path.basename(filename)[0] == 'n'):
                filename_without_ext = os.path.splitext(filename)[0]
                file_directory = DIRECTORY + os.sep + filename_without_ext

                signals_temp = wfdb.rdsamp(file_directory, channels=[0, 1], sampto=RECORD_DURATION_SECONDS*FREQUENCY_HERTZ)[0]
                signals_temp = signals_temp[np.newaxis,:,:]

                # fix ouliers in the last sample
                #LAST_SAMPLE = 230399
                #signals_temp[LAST_SAMPLE] = signals_temp[LAST_SAMPLE-1]

                # split records into 10 sec 
                #signals_temp = np.split(signals_temp, indices_or_sections=180)
                
                # fill resulting array
                signals = np.append(signals, signals_temp, axis=0)

                # assign patient number to 10 sec segments
                #patient_number = get_patient_number(filename)
                #patient_number_array_temp = np.full(180, patient_number, dtype=np.int32)
                #patient_number_array = np.append(patient_number_array, patient_number_array_temp)

                lables_temp = np.full(1, 0)
                labels = np.append(labels, lables_temp)


            else:
                pass

    return signals, labels, patient_number_array



DS_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/'

signals, labels, patient_number_array = load_dataset_whole_recordings(DS_DIRECTORY)

print(np.shape(signals))
print(np.shape(labels))
print(np.shape(patient_number_array))

########################################################PLOTTING#######################################################################
PLOTTING = True

if PLOTTING:
    # set record parameters
    RECORD_DURATION_SECONDS = 1800
    FREQUENCY_HERTZ = 128.0


    # from here only for plotting the signal
    nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
    t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

    

    # plot normalized signals
    signal_number = 2
    #print('Patient ID:')
    #print(patient_number_array[signal_number])
    print(labels[signal_number])

    #plt.figure(2)
    fig, axs = plt.subplots(2)

    z = signals[signal_number, :, 0]
    axs[0].plot(t, z)
    axs[0].set_title('lead 0', loc='left')
    axs[0].set_xlabel('time [seconds]')
    axs[0].set_ylabel('voltage [mV]')

    w = signals[signal_number, :, 1]
    axs[1].plot(t, w)
    axs[1].set_title('lead 1', loc='left')
    axs[1].set_xlabel('time [seconds]')
    axs[1].set_ylabel('voltage [mV]')


    plt.tight_layout()
    plt.show()

else:
    pass




########################################################PLOTTING#######################################################################