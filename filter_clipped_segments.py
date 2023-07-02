import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wfdb

from sklearn.model_selection import train_test_split

#from load_dataset_without_filter import load_dataset_PAF

#def clipping_filter(signals):

 #   clipping_value = np.bincount(signals).argmax()
 #   
 #   rows_to_delete = np.where(np.where(np.any(signals == clipping_value, axis=2)))[0]
 #   signals_fitered = np.delete(signals, rows_to_delete, axis=0)
 #   
 #   return signals_fitered


def clipping_filter_normalized_signal(signals_1, signals_2, labels):
    clipping_value = 1
    signals_stacked = np.stack((signals_1, signals_2), axis=2)

    rows_to_delete = np.where(np.where(np.any(signals_stacked == clipping_value, axis=2)))[0]
    signals_1_deleted = np.delete(signals_1, rows_to_delete, axis=0)
    signals_2_deleted = np.delete(signals_2, rows_to_delete, axis=0)
    labels_deleted = np.delete(labels, rows_to_delete, axis=0)

    return signals_1_deleted, signals_2_deleted, labels_deleted







if __name__ == '__main__':
    
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 5, 0],
                  [0, 0, 0, 0]]
                  )
    
    d = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]
                  )
    
    labels = np.array([0, 0, 1, 0])

    stack = np.stack((a, d), axis=0)
    print(stack)
    print(np.shape(a))
    print(np.shape(stack))
  
    rows = np.where(np.any(abs(stack) == 1, axis=0))[0]
    print(rows)

    #deleted = np.delete(stack, rows, axis=0)
    #print(deleted)
#
    #split_1, split_2 = np.split(deleted, 2, axis=0)
#
    #print(split_1)
    #print(split_2)


    a_deleted = np.delete(a, rows, axis=0)
    d_deleted = np.delete(d, rows, axis=0)

    print(a_deleted)
    print(d_deleted)


    a, d, labels =clipping_filter_normalized_signal(a, d, labels)

    print(a)
    print(d)
    print(labels)