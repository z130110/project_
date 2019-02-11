import sys
sys.path.insert(0, "/home/Workdir/bib/python_modeling/etl/transformations")
sys.path.insert(0, "/home/Workdir/bib/python_modeling/etl/load")
from data_loader import DataLoader
from process_useragents import ProcessUserAgents
from process_time import ProcessTime
from balance_dataset import BalanceDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections


def save_csv_to_np(data_dir, file_name):
    counts_for_each_target = 100
    save_dir = "/data/numpy_conversions/" + data_dir  + "/" + file_name[:-4] + "_" + str(counts_for_each_target) + "test"
    dl = features("data5", file_name, counts_for_each_target=counts_for_each_target)
    np.save(save_dir, dl)

def features(data_dir, file_name, counts_for_each_target=2000):
    dl = DataLoader(data_dir, file_name, features_to_use= \
        ["sessionstarttime_weekday", \
        "sessionstarttime_hour", \
        "data", \
        "os", \
        "browser", \
        "combined_pagelocation", \
        "combined_eventtimestamp", \
        "target"], counts_for_each_target=counts_for_each_target)

    t = time.time()
    print("After DataLoader row count,",len(dl.csv_data_dict["combined_pagelocation"]))
    print()

    print("processing time")
    processed_time = ProcessTime(dl.csv_data_dict["combined_pagelocation"], \
        dl.csv_data_dict["combined_eventtimestamp"]).run()
    processed_time_list = processed_time.tolist()
    t = time.time()
    print(time.time() - t)
    print()


    print("processing user agent")
    os, browser = ProcessUserAgents(dl.csv_data_dict["os"], \
        dl.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    t = time.time()
    print(time.time() - t)
    print()


    res_np = dl.to_numpy([dl.csv_data_dict["data"], \
        processed_time_list, \
        os, \
        browser, \
        dl.csv_data_dict["target"]])

    print("After DataLoader")
    print("Sum of all ages", sum(list(collections.Counter(res_np[:,-1]).values())))
    print(collections.Counter(res_np[:,-1]))
    print(sorted(list(collections.Counter(res_np[:,-1]).keys())))
    print()
    print("After BalanceDataset")
    bl = BalanceDataset(res_np, {"target" : -1}, "not classification").run()
    print("Sum of all ages", sum(list(collections.Counter(bl[:,-1]).values())))
    print(collections.Counter(bl[:,-1]))
    print(sorted(list(collections.Counter(bl[:,-1]).keys())))
    return bl


if __name__ == "__main__":
    #X_y = features("data4", "0.5_10_1.csv", counts_for_each_target=200)
    save_csv_to_np("data4", "0.5_10_1.csv")

    """
    save_csv_to_np("data4", "0.5_10_1.csv")
    save_csv_to_np("data4", "0.5_10_2.csv")
    save_csv_to_np("data4", "0.5_10_3.csv")
    save_csv_to_np("data4", "0.5_10_4.csv")
    save_csv_to_np("data4", "0.5_10_5.csv")
    save_csv_to_np("data4", "0.5_10_6.csv")
    """
