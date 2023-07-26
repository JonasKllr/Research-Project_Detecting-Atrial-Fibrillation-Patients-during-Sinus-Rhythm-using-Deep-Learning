import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import numpy as np
import os
import shutil
import wfdb
import wfdb.processing

from load_dataset_without_filter import get_header_comments_Dx_CinC

DIRECTORY_EVERYTHING = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/everything'
#DIRECTORY_EVERYTHING = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/test'
DIRECTORY_CLEANED = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_only'

for filename in sorted(os.listdir(DIRECTORY_EVERYTHING)):
    
    # only do data integration once per record
    if filename.endswith('.mat'):

        print(filename)
        
        filename_without_ext = os.path.splitext(filename)[0]
        file_dir_without_ext = DIRECTORY_EVERYTHING + os.sep + filename_without_ext

        comment_list = get_header_comments_Dx_CinC(file_dir_without_ext)
        SINUS_RHYTHM = '426783006'

        # load file only if it is labeled as sinus rhythm (426783006)
        if SINUS_RHYTHM in comment_list:

            file_dir_mat = DIRECTORY_EVERYTHING + os.sep + filename
            file_dir_hea = DIRECTORY_EVERYTHING + os.sep + filename_without_ext + '.hea'
            
            file_dir_mat_new = DIRECTORY_CLEANED + os.sep + filename
            file_dir_hea_new = DIRECTORY_CLEANED + os.sep + filename_without_ext + '.hea'

            shutil.copyfile(file_dir_mat, file_dir_mat_new)
            shutil.copyfile(file_dir_hea, file_dir_hea_new)

        else:
            pass

    else:
        pass



