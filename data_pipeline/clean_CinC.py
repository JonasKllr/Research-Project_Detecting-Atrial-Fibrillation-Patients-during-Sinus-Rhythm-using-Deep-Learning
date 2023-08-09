import sys
sys.path.append("/media/jonas/SSD_new/CMS/Semester_4/research_project/scripts")

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from load_dataset_without_filter import get_header_comments_Dx_CinC, get_header_comments_Age_CinC



def filter_sinus_rhythm():

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


def filter_no_age():
    
    DIRECTORY_SINUS_ONLY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_only'
    DIRECTORY_SINUS_AND_AGE_ONLY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_and_age_only'

    for filename in sorted(os.listdir(DIRECTORY_SINUS_ONLY)):
        if filename.endswith('.mat'):
            print(filename)

            filename_without_ext = os.path.splitext(filename)[0]
            file_dir_without_ext = DIRECTORY_SINUS_ONLY + os.sep + filename_without_ext
            age = get_header_comments_Age_CinC(file_dir_without_ext)

            if np.isnan(age):
                pass
            else:
                file_dir_mat = DIRECTORY_SINUS_ONLY + os.sep + filename
                file_dir_hea = DIRECTORY_SINUS_ONLY + os.sep + filename_without_ext + '.hea'

                file_dir_mat_new = DIRECTORY_SINUS_AND_AGE_ONLY + os.sep + filename
                file_dir_hea_new = DIRECTORY_SINUS_AND_AGE_ONLY + os.sep + filename_without_ext + '.hea'

                shutil.copyfile(file_dir_mat, file_dir_mat_new)
                shutil.copyfile(file_dir_hea, file_dir_hea_new)


def find_age_300():
    
    DIRECTORY_SINUS_AND_AGE_ONLY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_and_age_only'

    age = np.empty(0)

    for filename in sorted(os.listdir(DIRECTORY_SINUS_AND_AGE_ONLY)):
        if filename.endswith('.mat'):
            print(filename)

            filename_without_ext = os.path.splitext(filename)[0]
            file_dir_hea = DIRECTORY_SINUS_AND_AGE_ONLY + os.sep + filename_without_ext
            age_temp = get_header_comments_Age_CinC(file_dir_hea)

            age = np.append(age, age_temp)

            age_300 = np.empty(0)
            if age_temp > 200.0:
                age_300 = np.append(age_300, filename)
                #sys.exit()



    #np.savetxt('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/age_300.txt', age_300)

    print(age)
    print(np.shape(age))

    nan = np.isnan(age).any()
    print(nan)

    print(age.max())
    print(age.min())

    # histogram
    plt.hist(age, bins='auto')
    plt.show()



def filter_for_age_not_300():
    
    DIRECTORY_SINUS_AND_AGE_ONLY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_and_age_only'
    DIRECTORY_SINUS_AND_AGE_NOT_300_ONLY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/sinus_and_age-not-300_only'

    for filename in sorted(os.listdir(DIRECTORY_SINUS_AND_AGE_ONLY)):
        if filename.endswith('.mat'):
            print(filename)

            filename_without_ext = os.path.splitext(filename)[0]
            file_dir_without_ext = DIRECTORY_SINUS_AND_AGE_ONLY + os.sep + filename_without_ext
            age = get_header_comments_Age_CinC(file_dir_without_ext)


            if age > 200.0:
                pass
            else:
                file_dir_mat = DIRECTORY_SINUS_AND_AGE_ONLY + os.sep + filename
                file_dir_hea = DIRECTORY_SINUS_AND_AGE_ONLY + os.sep + filename_without_ext + '.hea'

                file_dir_mat_new = DIRECTORY_SINUS_AND_AGE_NOT_300_ONLY + os.sep + filename
                file_dir_hea_new = DIRECTORY_SINUS_AND_AGE_NOT_300_ONLY + os.sep + filename_without_ext + '.hea'

                shutil.copyfile(file_dir_mat, file_dir_mat_new)
                shutil.copyfile(file_dir_hea, file_dir_hea_new)



def get_age_distribution(DIRECTORY):

    age = np.empty(0)

    for filename in sorted(os.listdir(DIRECTORY)):
        if filename.endswith('.mat'):
            print(filename)

            filename_without_ext = os.path.splitext(filename)[0]
            file_dir_hea = DIRECTORY + os.sep + filename_without_ext
            age_temp = get_header_comments_Age_CinC(file_dir_hea)

            age = np.append(age, age_temp)

    
    np.savetxt('/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/age_array.txt', age, fmt='%d')

    # histogram
    plt.hist(age, bins='auto')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    plt.clf()

    plt.hist(age, bins=10)
    plt.title('Age grouped in 10 categories')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':

    dir = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/training_prepared/'

    array_file = dir + 'age_array.txt'
    age = np.loadtxt(array_file)

    labels = np.load(dir + 'CinC_labels_5.npy', allow_pickle=False)
    labels = labels.astype(int)

    print(age.max())

    plt.hist(age, bins=5 ,density=False)
    plt.title('Age grouped in 5 categories')
    plt.xlabel('Age [years]')
    plt.ylabel('Frequency [-]')
    plt.show()


