import numpy as np
import os

DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/history/'


for subdir, dirs, files in os.walk(DIR + 'history_ibmt/'):
    for file in files:
        if file.endswith('history.csv'):
            print(os.path.join(subdir, file))
            file_dir = os.path.join(subdir, file)

            print(subdir)

            with open(file_dir, 'r') as f:
                data = [line.replace('"', '') for line in f]
                data = [line.replace('[', '') for line in data]
                data = [line.replace(']', '') for line in data]
                data = [line.strip() for line in data]
                data = [line.split(',') for line in data]
            
            np.savetxt(subdir + '/history_cleaned.csv', data, delimiter=',', fmt='% s')






#with open(DIR + 'history.csv', 'r') as f:
#    data = [line.replace('"', '') for line in f]
#    data = [line.replace('[', '') for line in data]
#    data = [line.replace(']', '') for line in data]
#    data = [line.strip() for line in data]
#    data = [line.split(',') for line in data]



#np.savetxt(DIR + 'history_new.csv', data, delimiter=',', fmt='% s')