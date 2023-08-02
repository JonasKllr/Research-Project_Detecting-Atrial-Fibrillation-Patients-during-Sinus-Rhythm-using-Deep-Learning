import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'

signals = np.load(SAVE_DIR + 'CinC_signals.npy', allow_pickle=False)
labels = np.load(SAVE_DIR + 'CinC_labels.npy', allow_pickle=False)

print(labels)


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