from etl.transformations.process_time import ProcessTime
from etl.transformations.getavg_time import GetAvgTime
from etl.transformations.balance_dataset_class import BalanceDatasetClass
from etl.transformations.login_time import LoginTime
from etl.transformations.page_encode import PageEncode
from etl.transformations.normalize_time import NormalizeTime
from etl.transformations.process_useragents import ProcessUserAgents
from etl.load.data_loader import DataLoader
from sklearn.model_selection import train_test_split
from etl.transformations.process_time import ProcessTime
from etl.transformations.time_discretization import TimeDiscretize
from etl.transformations.bow import BOW

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections


def save_csv_to_np(data_dir, file_name, percent_to_read = 1):
    save_dir = "/data/numpy_conversions/data10/" + file_name[:-4] + "_" + "balanced_2_with_login_time"
    features(data_dir, file_name, percent_to_read)
    #print(dl.shape)
    #np.save(save_dir, dl)

def features(data_dir, file_name, percent_to_read):
#sessionnumber,devicebrowsername,deviceplatformname,devicesystemname,page_flow,time_list,target,browser_indexer,platform_indexer,OS_indexer,check_num_pages,word_vector_pagelist
    dl = DataLoader(data_dir, file_name, features_to_use=[
        "page_flow",
        "time_list",
        "browser_indexer",
        "platform_indexer",
        "OS_indexer",
        "check_num_pages",
        "word_vector_pagelist",
        "target"], percent_to_read=percent_to_read)
    t1(dl, file_name)

def t1(loaded_data, filename):
    save_dir = "/data/numpy_conversions/task3/data0/"

    #login1,login2,login3,login4 = LoginTime(loaded_data.csv_data_dict["sessionstarttime_hour"],
    #         loaded_data.csv_data_dict["sessionstarttime_minute"],
    #         loaded_data.csv_data_dict["sessionstarttime_weekday"]).run()

    avg_time, avg_std = GetAvgTime(loaded_data.csv_data_dict["time_list"]).run()
    avg_time = avg_time.tolist()
    avg_std = avg_std.tolist()

    res_np = loaded_data.to_numpy([loaded_data.csv_data_dict["word_vector_pagelist"],
        loaded_data.csv_data_dict["OS_indexer"],
        loaded_data.csv_data_dict["browser_indexer"],
        avg_time,
        avg_std,
        loaded_data.csv_data_dict["target"]])

    train, test = split_and_balance_data(res_np)
    filename = save_dir + "task3_" + filename[:-4]
    print(filename)
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)

def split_and_balance_data(np_arr):
    train, test = train_test_split(np_arr, test_size=0.25, random_state=42)
    print("Train set size", train.shape, "Test set size", test.shape)
    train_balanced = BalanceDatasetClass(train, {"target" : -1}, "not classification").run()
    return train_balanced, test

if __name__ == "__main__":
    save_csv_to_np("task3/data0", "10_1_.csv",0.5)
    save_csv_to_np("task3/data0", "10_2_.csv",0.5)
    save_csv_to_np("task3/data0", "10_5_.csv",0.5)
    save_csv_to_np("task3/data0", "15_3_.csv",0.5)
    save_csv_to_np("task3/data0", "20_1_.csv",0.5)
    save_csv_to_np("task3/data0", "20_2_.csv",0.5)
    save_csv_to_np("task3/data0", "20_5_.csv",0.5)
    save_csv_to_np("task3/data0", "30_1_.csv",0.5)
    save_csv_to_np("task3/data0", "30_2_.csv",0.5)
    save_csv_to_np("task3/data0", "50_1_.csv",0.5)
    save_csv_to_np("task3/data0", "50_2_.csv",0.5)
    save_csv_to_np("task3/data0", "50_5_.csv",0.5)
