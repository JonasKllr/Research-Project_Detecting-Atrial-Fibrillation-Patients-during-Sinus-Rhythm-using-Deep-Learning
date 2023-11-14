import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import numpy as np
import matplotlib.pyplot as plt


from data_pipeline.load_dataset_without_filter import load_dataset_PAF
from data_pipeline.filter_clipped_segments import find_clipped_segments, delete_clipped_segments
from data_pipeline.normalize import normalize_ecg

def less_segments(signals, labels, patient_number_array):
    patient_number_array_unique, value_counts = np.unique(patient_number_array, return_counts=True)

    # Number of elements you want to pick for each unique value
    num_elements_per_value = 35

    # Initialize a list to store selected elements
    selected_elements = []
    selected_signals = []
    selected_labels = []

    # Loop through each unique value
    for value, count in zip(patient_number_array_unique, value_counts):
        # Calculate the step size to select approximately 35 equally spaced elements
        step_size = max(1, count // num_elements_per_value)
        
        # Generate indices for selecting elements
        indices = np.arange(0, count, step_size)[:num_elements_per_value]
        
        # Offset the indices based on the unique value's position in the original array
        indices += np.where(patient_number_array == value)[0][0]
        
        # Select elements using the calculated indices
        selected_elements.extend(patient_number_array[indices])
        selected_signals.extend(signals[indices,:,:])
        selected_labels.extend(labels[indices])

    # Convert the selected elements to a NumPy array
    selected_elements = np.array(selected_elements)
    selected_signals = np.array(selected_signals)
    selected_labels = np.array(selected_labels)

    return selected_signals, selected_labels, selected_elements


if __name__ == '__main__':
    # load data
    FILE_DIRECTORY = "/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/"
    signals, labels, patient_number_array = load_dataset_PAF(FILE_DIRECTORY)

    # data normalization into range [0.0, 1.0] to filter for signal clipping
    signals_norm = normalize_ecg(signals)

    #find rows in signals array with clipped segments
    clipped_segments = find_clipped_segments(signals_norm)
    del signals_norm

    # delete the segments containing signal clipping
    signals, labels, patient_number_array = delete_clipped_segments(signals, labels, patient_number_array, clipped_segments)
    patient_number_array_unique = np.unique(patient_number_array, return_index=False)

    print(np.shape(signals))
    print(np.shape(labels))
    print(np.shape(patient_number_array))
    print(np.shape(patient_number_array_unique))

    signals, labels, patient_number_array =  less_segments(signals, labels, patient_number_array)

    patient_number_array_unique = np.unique(patient_number_array, return_index=False)

    print(np.shape(signals))
    print(np.shape(labels))
    print(np.shape(patient_number_array))
    print(np.shape(patient_number_array_unique))


    ########################################################PLOTTING#######################################################################
    PLOTTING = True

    if PLOTTING:
        # set record parameters
        RECORD_DURATION_SECONDS = 10
        FREQUENCY_HERTZ = 128.0


        # from here only for plotting the signal
        nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
        t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

        #plt.figure(2)
        fig, axs = plt.subplots(2)

        # plot normalized signals
        signal_number = 800
        print(labels[signal_number])

        z = signals[signal_number, :, 0]
        axs[0].plot(t, z)
        axs[0].set_title('lead 0 normalized', loc='left')
        axs[0].set_xlabel('seconds')

        w = signals[signal_number, :,1]
        axs[1].plot(t, w)
        axs[1].set_title('lead 1 normalized', loc='left')
        axs[1].set_xlabel('seconds')


        plt.tight_layout()
        plt.show()

    else:
        pass




    ########################################################PLOTTING#######################################################################