import sys
sys.path.append("/home/joke793c/research_project/scripts")

import numpy as np



def normalize_ecg(signals):
    # get min max value in every row
    min_val_0 = signals[:,:,0].min(axis=1)
    max_val_0 = signals[:,:,0].max(axis=1)
    
    min_val_1 = signals[:,:,1].min(axis=1)
    max_val_1 = signals[:,:,1].max(axis=1)

    # add another dimension to enable broadcasting
    min_val_0 = min_val_0[:, np.newaxis]
    max_val_0 = max_val_0[:, np.newaxis]
    
    min_val_1 = min_val_1[:, np.newaxis]
    max_val_1 = max_val_1[:, np.newaxis]

    # normalize signals: y = (x - min(x)) / (max(x) - min(x))
    signals_norm = np.ones_like(signals)
    np.divide((signals[:,:,0]- min_val_0), (max_val_0 - min_val_0), out=signals_norm[:,:,0], where=(max_val_0 - min_val_0) != 0)
    np.divide((signals[:,:,1]- min_val_1), (max_val_1 - min_val_1), out=signals_norm[:,:,1], where=(max_val_1 - min_val_1) != 0)
    

    return signals_norm



if __name__ == '__main__':
    
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 5, 0],
                  [0, 0, 1, 0]]
                  )
    
    d = np.array([[0.5, 2, 1, 0],
                  [0, 0, 0, 0],
                  [1, 0, -1, 0]]
                  )
    
    labels = np.array([0, 1, 0])

    stack = np.stack((a, d), axis=2)

    a_norm = normalize_ecg(stack)
    print(a_norm)
    print(a_norm.max())
    print(a_norm.min())










    # find min and max value in every row
    #min_val = a.min(axis=1)
    #max_val = a.max(axis=1)

    # add another axis to enable broadcasting
    #min_val = min_val[:, np.newaxis]
    #max_val = max_val[:, np.newaxis]

    #print(min_val)
    #print(max_val)

    # normalize
    #a_norm = (a - min_val) / (max_val - min_val)

    #print(a_norm)

