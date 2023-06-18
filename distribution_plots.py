import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_plots(data):
    '''
    Returns and displays the bar plots for categorical features and histograms for continuous-valued features
    
    Parameters:
    data: pd.DataFrame object
    
    Returns:
    Distribution plots of all the columns in the dataframe
    '''
    nrows, ncols = data.shape
    fig = plt.figure(figsize=(int(1.5*ncols), ncols))
    cols = 5
    rows = np.ceil(float(ncols) / cols)
    for i, column in enumerate(data.columns):
        ax = fig.add_subplot(int(rows), cols, i + 1)
        ax.set_title(column)
        if data.dtypes[column] == object:
            data[column].value_counts().plot(kind="bar", axes=ax)
        else:
            data[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
