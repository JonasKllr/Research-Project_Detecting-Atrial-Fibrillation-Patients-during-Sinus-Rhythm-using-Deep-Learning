import matplotlib.pyplot as plt
import numpy as np

from filter_clipped_segments import clipping_filter_normalized_signal_sliding_window


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
    #array_stacked = abs(array_stacked)
    print(np.shape(array_stacked))

    # absulut value to check for negative and positive clipping
    #a = abs(a)
    #print(a)

    window_shape = 3
    clipping_window = np.array([1, 1, 1])
    print(np.shape(clipping_window)[0])
    # window_view = np.lib.stride_tricks.sliding_window_view(a, window_shape=window_shape, axis=1)
    window_view = np.lib.stride_tricks.sliding_window_view(array_stacked, window_shape=window_shape, axis=1)
    print(window_view)
    print(np.shape(window_view))


    #rows_to_delete = np.where((np.all(np.equal(abs(window_view), clipping_window), axis=2)))[0]
    rows_to_delete = np.where((np.all(np.equal(abs(window_view), clipping_window), axis=3)))[0]
    print(rows_to_delete)

    #a_deleted = np.delete(a, rows_to_delete, axis=0)
    #print(a_deleted)

    array_stacked_deleted = np.delete(array_stacked, rows_to_delete, axis=0)
    print(array_stacked_deleted)

    signals, labels = clipping_filter_normalized_signal_sliding_window(array_stacked, labels)
    print(signals)
    print(labels)
    
    
    
