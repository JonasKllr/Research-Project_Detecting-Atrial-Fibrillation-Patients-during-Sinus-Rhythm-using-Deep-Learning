# code for filter taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
import numpy as np
import wfdb

from scipy.signal import butter, lfilter


def butter_bandpass(LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER):
    nyq = 0.5 * FREQUENCY_HERTZ
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(ORDER, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER=ORDER)
    return lfilter(b, a, data)





# TESTS
if __name__ == '__main__':
    # set record parameters
    RECORD_DURATION_SECONDS = 10
    FREQUENCY_HERTZ = 128.0

    # set filter parameters
    LOWCUT = 0.3
    HIGHCUT = 50.0
    ORDER = 5   # value 8 taken from paper (but might be too high for my usecase)

    #set path 
    ONE_SECOND = 128
    record = wfdb.rdrecord('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/physionet.org/files/afpdb/cleaned/n03',sampfrom=int(0), sampto=int(RECORD_DURATION_SECONDS*ONE_SECOND), channels=[0])



    # from here only for plotting the signal
    nsamples = int(RECORD_DURATION_SECONDS * FREQUENCY_HERTZ)
    t = np.linspace(0, RECORD_DURATION_SECONDS, nsamples, endpoint=False)

    x = record.p_signal


    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Original signal')

    y = butter_bandpass_filter(x, LOWCUT, HIGHCUT, FREQUENCY_HERTZ, ORDER)
    print(np.shape(y))
    print(type(y))

    plt.plot(t, y, label='Filtered signal')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    SAFE_DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/figures/paf_pred_challenge/'
    #plt.savefig(SAFE_DIRECTORY + 'n01_filtered.png')

    plt.show()
