import numpy as np





def clipping_filter_normalized_signal(signals, labels):
    # set content of window to be searched for
    WINDOW_SIZE = 6
    clipping_window_0 = np.full((WINDOW_SIZE,), 0.0)
    clipping_window_1 = np.full((WINDOW_SIZE,), 1.0)
    
    # slide window over signals and get rows where CLIPPING_WINDOW and signals match
    window_view = np.lib.stride_tricks.sliding_window_view(signals, window_shape=WINDOW_SIZE, axis=1)
    rows_to_delete_0 = np.where((np.all(np.equal(abs(window_view), clipping_window_0), axis=3)))[0]
    rows_to_delete_1 = np.where((np.all(np.equal(abs(window_view), clipping_window_1), axis=3)))[0]
    rows_to_delete = np.concatenate([rows_to_delete_0, rows_to_delete_1])
    
    signals_deleted = np.delete(signals, rows_to_delete, axis=0)
    labels_deleted = np.delete(labels, rows_to_delete, axis=0)

    return signals_deleted, labels_deleted


def find_clipped_segments(signals):

    print('Searching for clipped segments ...')

    # content of window to be searched for
    WINDOW_SIZE = 8
    clipping_window_0 = np.full((WINDOW_SIZE,), 0.0)
    clipping_window_1 = np.full((WINDOW_SIZE,), 1.0)
    
    # slide window over signals and get rows where CLIPPING_WINDOW and signals match
    window_view = np.lib.stride_tricks.sliding_window_view(signals, window_shape=WINDOW_SIZE, axis=1)
    rows_to_delete_0 = np.where((np.all(np.equal(abs(window_view), clipping_window_0), axis=3)))[0]
    rows_to_delete_1 = np.where((np.all(np.equal(abs(window_view), clipping_window_1), axis=3)))[0]

    return np.concatenate([rows_to_delete_0, rows_to_delete_1])


def delete_clipped_segments(signals, labels, ROWS_TO_DELETE):
    
    print('Deleting clipped segments ...')

    signals_deleted = np.delete(signals, ROWS_TO_DELETE, axis=0)
    labels_deleted = np.delete(labels, ROWS_TO_DELETE, axis=0)

    return signals_deleted, labels_deleted







if __name__ == '__main__':
    
    a = np.array([[0, 0, 1, 1, 3],
                  [0, 0, 5, 0, 3],
                  [0, 0, 1, 0, 3]]
                  )
    
    d = np.array([[0, 0, 1, 0, 5],
                  [0, 0, 0, 0, 5],
                  [1, 0, 0, 0, 5]]
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