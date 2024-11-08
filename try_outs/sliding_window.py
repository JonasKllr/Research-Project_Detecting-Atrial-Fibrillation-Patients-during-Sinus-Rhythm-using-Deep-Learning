import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.filter_clipped_segments import clipping_filter_normalized_signal_sliding_window


if __name__ == '__main__':

    a = np.array([[-1, 1, 0, 0],
                  [0, 0, 5, 0],
                  [0, 1, 1, 1],
                  [0, 0, 1, 1],
                  [1, 0, 0, 0]]
                  )
    
    b = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 1, 1, 0]]
                  )
    
    labels = np.array([0, 1, 0, 0, 1])

    array_stacked = np.stack((a, b), axis=2)
    print(np.shape(array_stacked))

    window_shape = 3
    clipping_window = np.array([1, 1, 1])
    print(np.shape(clipping_window)[0])

    window_view = np.lib.stride_tricks.sliding_window_view(array_stacked, window_shape=window_shape, axis=1)
    print(window_view)
    print(np.shape(window_view))

    rows_to_delete = np.where((np.all(np.equal(abs(window_view), clipping_window), axis=3)))[0]
    print(rows_to_delete)

    array_stacked_deleted = np.delete(array_stacked, rows_to_delete, axis=0)
    print(array_stacked_deleted)

    signals, labels = clipping_filter_normalized_signal_sliding_window(array_stacked, labels)
    print(signals)
    print(labels)
    
    
    
