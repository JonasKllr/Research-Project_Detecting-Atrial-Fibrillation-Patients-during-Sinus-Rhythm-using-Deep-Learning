import os
import wfdb


# Get comments from header files under 'Dx'. Labels of the recording.
def get_header_comments_Dx_CinC(RECORD_DIR):
    
    header = wfdb.rdheader(RECORD_DIR)
    header_comments = header.comments
    
    # get entry under 'Dx' and convert into list with one label per element.
    comment_list = list()
    for comment in header_comments:
        if comment.startswith('Dx'):
            try:
                entries = comment.split(': ')[1].split(',')
                for entry in entries:
                    comment_list.append(entry.strip())
            except:
                pass
    
    return comment_list



if __name__ == '__main__':

    DIRECTORY = '/media/jonas/SSD_new/CMS/Semester_4/research_project/datasets/CinC/test'

    for filename in sorted(os.listdir(DIRECTORY)):
        
        # only do data integration once per record
        if filename.endswith('.mat'):
            
            filename_without_ext = os.path.splitext(filename)[0]
            header_file_dir = DIRECTORY + os.sep + filename_without_ext

            comment_list = get_header_comments_Dx_CinC(header_file_dir)
            print(comment_list)

            SINUS_RHYTHM = '426783006'

            if SINUS_RHYTHM in comment_list:
                print('yes')
            else:
                print('no')

        

        

        
            


