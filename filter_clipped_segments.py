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


def clipping_filter_normalized_signal(signals, labels):
    clipping_value = 1.0
    #signals_stacked = np.stack((signals_1, signals_2), axis=2)

    rows_to_delete = np.where((np.any(abs(signals) == clipping_value, axis=2)))[0]
    print('rows_to_delete')
    print(rows_to_delete)
    print(np.shape(rows_to_delete))

    signals_deleted = np.delete(signals, rows_to_delete, axis=0)
    #signals_2_deleted = np.delete(signals_2, rows_to_delete, axis=0)
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