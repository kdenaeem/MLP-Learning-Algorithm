import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def read_data():
    dataset = pd.read_csv("standardised_data.csv")

    diego_temp_data = dataset.loc[:,
                    ['Date', 'T', 'W', 'SR', 'DSP', 'DRH', 'PanE', 'Index']]

    output_values = diego_temp_data

    return output_values




def clean_data(diego_temp_data):

    # this loop ensures all values in dataframe are numeric.
    # if not, "coerce" sets the invalid data to NaN
    for col in diego_temp_data.columns:
        diego_temp_data[col] = pd.to_numeric(diego_temp_data[col], errors='coerce')

    # drops NaN values from dataframe
    diego_temp_data.dropna(axis=0, inplace=True)
    # ensures all values are positive
    df_flood_data = diego_temp_data[(diego_temp_data > 0).all(axis=1)]
    # converts dataFrame to csv file
    df_flood_data.to_csv("clean_data.csv", index=False)

def standardise_data(diego_temp_data):
    # dictionary to find the max and minimum values for each column
    col_dict = {}
    for col in diego_temp_data.columns:
        col_dict[col] = [diego_temp_data[col].max(), diego_temp_data[col].min()]

    # standardising each value in dataframe and appending to dictionary
    standardised_dict = {}
    # loops through each column in dataframe
    for col in diego_temp_data:
        column_data = diego_temp_data[col]
        standardised_dict[col] = []
        # loops through each value in the column
        # run the standardisation formula on each value in the column
        for val in column_data:
            s = 0.8 * ((val - col_dict[col][1]) / (col_dict[col][0] - col_dict[col][1])) + 0.1
            standardised_dict[col].append(s)

    # create a new dataframe of standardised data

    standardised_df = pd.DataFrame.from_dict(standardised_dict)
    # writing the dataframe to csv file
    standardised_df.to_csv("standardised_data.csv", index=False)
    return standardised_df

def split_data():
    df = pd.read_csv('standardised_data.csv')
    # random_state parameter is set to 42 to ensure that the same random split is obtained every time the code is run.

    train, val_test = train_test_split(df, test_size=0.6, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)

    data_train = train
    x1 = data_train[["T", "W", "SR", "DSP", "DSP"]].to_numpy()
    y1 = data_train[["PanE"]].to_numpy()

    data_val = val
    x1_val = data_val[["T", "W", "SR", "DSP", "DSP"]].to_numpy()
    y1_val = data_val[["PanE"]].to_numpy()
    # print("\n\n\nTHIS IS X1", x1.shape,"\n\n\n\nThis is y1", y1.shape)
    # print(y1)
    # print("this is x1 val\n\n\n", x1_val,"this is y1 val\n\n\n", y1_val)

    return x1, y1, x1_val, y1_val

split_data()


data = read_data()
# fig, ax = plt.subplots()
#
# x = data['T']
# y = data['PanE']
# plt.scatter(x, y)
# ax.set_title('PanE vs. T')
# ax.set_xlabel('T')
# ax.set_ylabel('PanE')
# plt.show()


