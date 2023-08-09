import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def raw_results_into_dataframe(DIR):

    # header for dataframe
    best_values = [['model', 'kernel_size', 'pooling_layer', 'learning_rate', 'fold', 'epoch', 'binary_accuracy', 'f1_score', 'loss', 'val_binary_accuracy', 'val_f1_score', 'val_loss']]
    
    # get information from file path and history.csv
    for subdir, dir, files in os.walk(DIR + 'history_ibmt/'): # history_taurus/history/, history_ibmt/
        for file in files:
            
            if file.endswith('history_cleaned.csv'):
                
                file_dir = os.path.join(subdir, file)
                subdir = os.path.normpath(subdir)
                subdir = subdir.split(os.sep)

                row_temp = []
                # get model
                if subdir[len(subdir)-5] == "Model_1-blocks_3-layers_per_block_2":
                    row_temp.append('model_1')
                elif subdir[len(subdir)-5] == "Model_2-blocks_3-layers_per_block_1":
                    row_temp.append('model_2')
                elif subdir[len(subdir)-5] == "Model_3-blocks_2-layers_per_block_2":
                    row_temp.append('model_3')
                elif subdir[len(subdir)-5] == "Model_4-blocks_2-layers_per_block_1":
                    row_temp.append('model_4')
                else:
                    pass

                # get kernels size
                if subdir[len(subdir)-4] == "kernel_3":
                    row_temp.append(3)
                elif subdir[len(subdir)-4] == "kernel_6":
                    row_temp.append(6)
                elif subdir[len(subdir)-4] == "kernel_9":
                    row_temp.append(9)
                elif subdir[len(subdir)-4] == "kernel_12":
                    row_temp.append(12)
                else:
                    pass

                # get pooling layer
                if subdir[len(subdir)-3] == "pooling_max_pool":
                    row_temp.append('max_pool')
                elif subdir[len(subdir)-3] == "pooling_avg_pool":
                    row_temp.append('avg_pool')
                else:
                    pass

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
                index_min = train_result['val_loss'].idxmin()
                row_df_temp = train_result.loc[index_min, :].values.flatten().tolist()
                row_temp = row_temp + row_df_temp

                best_values.append(row_temp)

    # convert list into pandas dataframe
    best_values_df = pd.DataFrame(best_values)
    best_values_df = best_values_df.rename(columns=best_values_df.iloc[0]).loc[1:]
    #print(best_values_df.head(24))

    return best_values_df


def mean_dataframe(df: pd.DataFrame):

    mean_results = df.groupby(['model', 'kernel_size', 'pooling_layer', 'learning_rate']).mean()
    #print(mean_results.head(24))

    return mean_results


def median_dataframe(df: pd.DataFrame):

    median_results = df.groupby(['model', 'kernel_size', 'pooling_layer', 'learning_rate']).median()
    #print(median_results.head(12))

    return median_results


if __name__ == '__main__':

    DIR = '/media/jonas/SSD_new/CMS/Semester_4/research_project/history/'

    raw_df = raw_results_into_dataframe(DIR)

    mean_results = mean_dataframe(raw_df)
    median_results = median_dataframe(raw_df)
    mean_results = mean_results[mean_results.columns].astype(float) 
    median_results = median_results[median_results.columns].astype(float) 

    x_label_list = ['KS 3 - PL avg', 'KS 3 - PL max', 'KS 6 - PL avg', 'KS 6 - PL max', 'KS 9 - PL avg', 'KS 9 - PL max', 'KS 12 - PL avg', 'KS 12 - PL max']
    y_label_list = ['Model 1 - LR 0.0001', 'Model 1 - LR 0.001', 'Model 1 - LR 0.01', 'Model 2 - LR 0.0001', 'Model 2 - LR 0.001', 'Model 2 - LR 0.01', 'Model 3 - LR 0.0001', 'Model 3 - LR 0.001', 'Model 3 - LR 0.01', 'Model 4 - LR 0.0001', 'Model 4 - LR 0.001', 'Model 4 - LR 0.01']


    fig, ax = plt.subplots()
    sns.heatmap(mean_results.pivot_table(
        values='val_binary_accuracy',
        index=['model', 'learning_rate'],
        columns=['kernel_size', 'pooling_layer']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5, xticklabels=x_label_list, yticklabels=y_label_list)   # mako_r, crest

    ax.set_title('Mean Accuracy', weight='bold')
    ax.set_xlabel('Kernel Size (KS) - Pooling Layer (PL)', weight='bold')
    ax.set_ylabel('Model - Learning Rate (LR)', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(median_results.pivot_table(
        values='val_binary_accuracy',
        index=['model', 'learning_rate'],
        columns=['kernel_size', 'pooling_layer']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5, xticklabels=x_label_list, yticklabels=y_label_list)   # mako_r, crest

    ax.set_title('Median Accuracy', weight='bold')
    ax.set_xlabel('Kernel Size (KS) - Pooling Layer (PL)', weight='bold')
    ax.set_ylabel('Model - Learning Rate (LR)', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(mean_results.pivot_table(
        values='val_f1_score',
        index=['model', 'learning_rate'],
        columns=['kernel_size', 'pooling_layer']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5, xticklabels=x_label_list, yticklabels=y_label_list)   # mako_r, crest

    ax.set_title('Mean f1-Score', weight='bold')
    ax.set_xlabel('Kernel Size (KS) - Pooling Layer (PL)', weight='bold')
    ax.set_ylabel('Model - Learning Rate (LR)', weight='bold')
    plt.tight_layout()


    fig, ax = plt.subplots()
    sns.heatmap(median_results.pivot_table(
        values='val_f1_score',
        index=['model', 'learning_rate'],
        columns=['kernel_size', 'pooling_layer']
    ), annot=True ,cmap='mako_r', cbar=1, linewidths=0.5, xticklabels=x_label_list, yticklabels=y_label_list)   # mako_r, crest

    ax.set_title('Median f1-Score', weight='bold')
    ax.set_xlabel('Kernel Size (KS) - Pooling Layer (PL)', weight='bold')
    ax.set_ylabel('Model - Learning Rate (LR)', weight='bold')
    plt.tight_layout()


    plt.show()




  



            





