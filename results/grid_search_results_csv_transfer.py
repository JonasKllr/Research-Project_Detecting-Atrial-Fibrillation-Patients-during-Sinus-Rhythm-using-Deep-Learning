import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def raw_results_into_dataframe(DIR):

    # header for dataframe
    best_values = [['num_trained_layers', 'learning_rate', 'fold', 'epoch', 'binary_accuracy', 'f1_score', 'loss', 'val_binary_accuracy', 'val_f1_score', 'val_loss']]
    
    # get information from file path and history.csv
    for subdir, dir, files in os.walk(DIR): # history_taurus/history/, history_ibmt/
        for file in files:
            
            if file.endswith('history_cleaned.csv'):
                
                file_dir = os.path.join(subdir, file)
                subdir = os.path.normpath(subdir)
                subdir = subdir.split(os.sep)

                row_temp = []

                # get num trained layers
                if subdir[len(subdir)-3] == "Model_3_transfer_learning_1":
                    row_temp.append(1)
                elif subdir[len(subdir)-3] == "Model_3_transfer_learning_2":
                    row_temp.append(2)
                elif subdir[len(subdir)-3] == "Model_3_transfer_learning_3":
                    row_temp.append(3)
                elif subdir[len(subdir)-3] == "Model_3_transfer_learning_4":
                    row_temp.append(4)
                elif subdir[len(subdir)-3] == "Model_3_transfer_learning_5":
                    row_temp.append(5)
                elif subdir[len(subdir)-3] == "Model_3_transfer_learning_6":
                    row_temp.append(6)
                else:
                    pass

                # get learning rate
                if subdir[len(subdir)-2] == "learning_rate_0.01":
                    row_temp.append(0.01)
                elif subdir[len(subdir)-2] == "learning_rate_0.001":
                    row_temp.append(0.001)
                elif subdir[len(subdir)-2] == "learning_rate_0.0001":
                    row_temp.append(0.0001)
                elif subdir[len(subdir)-2] == "learning_rate_0.00001":
                    row_temp.append(0.00001)
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
                index_min = train_result['val_f1_score'].idxmax()
                row_df_temp = train_result.loc[index_min, :].values.flatten().tolist()
                row_temp = row_temp + row_df_temp

                best_values.append(row_temp)

    # convert list into pandas dataframe
    best_values_df = pd.DataFrame(best_values)
    best_values_df = best_values_df.rename(columns=best_values_df.iloc[0]).loc[1:]
    #print(best_values_df.head(24))

    return best_values_df


def mean_dataframe(df: pd.DataFrame):

    mean_results = df.groupby(['num_trained_layers', 'learning_rate']).mean()
    #print(mean_results.head(24))

    return mean_results


def median_dataframe(df: pd.DataFrame):

    median_results = df.groupby(['num_trained_layers', 'learning_rate']).median()
    #print(median_results.head(12))

    return median_results

def std_dataframe(df: pd.DataFrame):

    std_results = df.groupby(['num_trained_layers', 'learning_rate']).std()
    #print(mean_results.head(24))

    return std_results


if __name__ == '__main__':

    DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/history/transfer_learning/history/'

    raw_df = raw_results_into_dataframe(DIR)
    print(raw_df.head(90))
    
    # print specific rows
    print(raw_df.loc[raw_df['num_trained_layers'] == 5])

    mean_results = mean_dataframe(raw_df)
    median_results = median_dataframe(raw_df)
    std_results = std_dataframe(raw_df)
    mean_results = mean_results[mean_results.columns].astype(float) 
    median_results = median_results[median_results.columns].astype(float)
    std_results = std_results[std_results.columns].astype(float)

    mean_minus_median = mean_results.subtract(std_results)

    #print(std_results.head(24))

    #x_label_list = ['KS 3 - PL avg', 'KS 3 - PL max', 'KS 6 - PL avg', 'KS 6 - PL max', 'KS 9 - PL avg', 'KS 9 - PL max', 'KS 12 - PL avg', 'KS 12 - PL max']
    #y_label_list = ['Model 1 - LR 0.0001', 'Model 1 - LR 0.001', 'Model 1 - LR 0.01', 'Model 2 - LR 0.0001', 'Model 2 - LR 0.001', 'Model 2 - LR 0.01', 'Model 3 - LR 0.0001', 'Model 3 - LR 0.001', 'Model 3 - LR 0.01', 'Model 4 - LR 0.0001', 'Model 4 - LR 0.001', 'Model 4 - LR 0.01']


    fig, ax = plt.subplots()
    sns.heatmap(mean_results.pivot_table(
        values='val_binary_accuracy',
        index=['learning_rate'],
        columns=['num_trained_layers']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5)   # mako_r, crest

    ax.set_title('Mean Accuracy', weight='bold')
    ax.set_xlabel('Number Trained Layers', weight='bold')
    ax.set_ylabel('Learning Rate', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(mean_minus_median.pivot_table(
        values='val_binary_accuracy',
        index=['learning_rate'],
        columns=['num_trained_layers']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5)   # mako_r, crest

    ax.set_title('(Mean - StdDev) Accuracy', weight='bold')
    ax.set_xlabel('Number Trained Layers', weight='bold')
    ax.set_ylabel('Learning Rate', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(median_results.pivot_table(
        values='val_binary_accuracy',
        index=['learning_rate'],
        columns=['num_trained_layers']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5)   # mako_r, crest

    ax.set_title('Median Accuracy', weight='bold')
    ax.set_xlabel('Number Trained Layers', weight='bold')
    ax.set_ylabel('Learning Rate', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(mean_results.pivot_table(
        values='val_f1_score',
        index=['learning_rate'],
        columns=['num_trained_layers']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5)   # mako_r, crest

    ax.set_title('Mean f1-Score', weight='bold')
    ax.set_xlabel('Number Trained Layers', weight='bold')
    ax.set_ylabel('Learning Rate', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(mean_minus_median.pivot_table(
        values='val_f1_score',
        index=['learning_rate'],
        columns=['num_trained_layers']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5)   # mako_r, crest

    ax.set_title('(Mean - StdDev) f1-Score', weight='bold')
    ax.set_xlabel('Number Trained Layers', weight='bold')
    ax.set_ylabel('Learning Rate', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(median_results.pivot_table(
        values='val_f1_score',
        index=['learning_rate'],
        columns=['num_trained_layers']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5)   # mako_r, crest

    ax.set_title('Median f1-Score', weight='bold')
    ax.set_xlabel('Number Trained Layers', weight='bold')
    ax.set_ylabel('Learning Rate', weight='bold')
    plt.tight_layout()


    plt.show()




  



            





