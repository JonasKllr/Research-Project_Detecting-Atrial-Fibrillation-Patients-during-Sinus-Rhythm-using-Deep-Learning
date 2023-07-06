import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from sklearn.model_selection import train_test_split



def clipping_filter_normalized_signal(signals, labels):
    CLIPPING_WINDOW = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    window_view = np.lib.stride_tricks.sliding_window_view(signals, window_shape=np.shape(CLIPPING_WINDOW)[0], axis=1)
    rows_to_delete = np.where((np.all(np.equal(abs(window_view), CLIPPING_WINDOW), axis=3)))[0]
    
    signals_deleted = np.delete(signals, rows_to_delete, axis=0)
    labels_deleted = np.delete(labels, rows_to_delete, axis=0)

    return signals_deleted, labels_deleted



if __name__ == '__main__':
    
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 5, 0],
                  [0, 0, 1, 0]]
                  )
    
    d = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0]]
                  )
    
    labels = np.array([0, 1, 0])

    stack = np.stack((a, d), axis=2)
    #print(stack)
    #print(np.shape(a))
    print(np.shape(stack))
    print(stack)
  
    rows = np.where(np.any(abs(stack) == 1, axis=0))[0]
    #print(rows)

    #deleted = np.delete(stack, rows, axis=0)
    #print(deleted)
#
    #split_1, split_2 = np.split(deleted, 2, axis=0)
#
    #print(split_1)
    #print(split_2)


    #a_deleted = np.delete(a, rows, axis=0)
    #d_deleted = np.delete(d, rows, axis=0)

    #print(a_deleted)
    #print(d_deleted)


    stack, labels =clipping_filter_normalized_signal(stack, labels)

    print(stack)
    print(labels)
    print(np.shape(stack))
    print(np.shape(labels))