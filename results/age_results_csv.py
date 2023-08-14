import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def raw_results_into_dataframe(DIR):

    # header for dataframe
    best_values = [['learning_rate', 'fold', 'epoch', 'loss (MAE)', 'mean_squared_error',  'val_loss (MAE)', 'val_mean_squared_error']]
    
    # get information from file path and history.csv
    for subdir, dir, files in os.walk(DIR): 
        for file in files:
            
            if file.endswith('history.csv'):
                
                file_dir = os.path.join(subdir, file)
                subdir = os.path.normpath(subdir)
                subdir = subdir.split(os.sep)

                row_temp = []

                # get learning rate
                if subdir[len(subdir)-2] == "learning_rate_0.01":
                    row_temp.append(0.01)
                elif subdir[len(subdir)-2] == "learning_rate_0.001":
                    row_temp.append(0.001)
                elif subdir[len(subdir)-2] == "learning_rate_0.0001":
                    row_temp.append(0.0001)
                else:
                    pass

                # get fold number
                if subdir[len(subdir)-1] == "fold_1":
                    row_temp.append(1)
                elif subdir[len(subdir)-1] == "fold_2":
                    row_temp.append(2)
                elif subdir[len(subdir)-1] == "fold_3":
                    row_temp.append(3)
                elif subdir[len(subdir)-1] == "fold_4":
                    row_temp.append(4)
                elif subdir[len(subdir)-1] == "fold_5":
                    row_temp.append(5)
                else:
                    pass

                # get results from individual trainig
                train_result = pd.read_csv(file_dir)

                # get row where val_loss is min
                index_min = train_result['val_mean_squared_error'].idxmin()   # 'val_mean_squared_error', 'val_loss'
                row_df_temp = train_result.loc[index_min, :].values.flatten().tolist()
                row_temp = row_temp + row_df_temp

                best_values.append(row_temp)

    # convert list into pandas dataframe
    best_values_df = pd.DataFrame(best_values)
    best_values_df = best_values_df.rename(columns=best_values_df.iloc[0]).loc[1:]
    #print(best_values_df.head(24))

    return best_values_df


def mean_dataframe(df: pd.DataFrame):

    mean_results = df.groupby(['learning_rate']).mean()
    #print(mean_results.head(24))

    return mean_results


def median_dataframe(df: pd.DataFrame):

    median_results = df.groupby(['learning_rate']).median()
    #print(median_results.head(12))

    return median_results


if __name__ == '__main__':

    DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/history/age/wrong/history/Model_age_regression/kernel_6'

    raw_df = raw_results_into_dataframe(DIR)
    print(raw_df.head(20))

    mean_results = mean_dataframe(raw_df)
    median_results = median_dataframe(raw_df)
    mean_results = mean_results[mean_results.columns].astype(float) 
    median_results = median_results[median_results.columns].astype(float) 

    print(mean_results.head(5))
    print(median_results.head(5))



  



            





