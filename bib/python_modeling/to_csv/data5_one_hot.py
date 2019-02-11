import sys
sys.path.insert(0, "/home/BC2830/Celebrus/code/bib/python_modeling/etl/transformations")
sys.path.insert(0, "/home/BC2830/Celebrus/code/bib/python_modeling/etl/load")
from data_loader import DataLoader
from process_useragents import ProcessUserAgents
from process_time import ProcessTime
from balance_dataset import BalanceDataset
from page_encode import PageEncode

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections


def save_csv_to_np(data_dir, file_name):
    counts_for_each_target = 400
    save_dir = "/data/numpy_conversions/" + data_dir  + "/" + file_name[:-4] + "_" + str(counts_for_each_target)
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

    page = PageEncode(dl.csv_data_dict["combined_pagelocation"]).run()
    """
    processed_time = ProcessTime(dl.csv_data_dict["combined_pagelocation"], \
        dl.csv_data_dict["combined_eventtimestamp"]).run()
    processed_time_list = processed_time.tolist()
    t = time.time()


    os, browser = ProcessUserAgents(dl.csv_data_dict["os"], \
        dl.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    t = time.time()

    res_np = dl.to_numpy([dl.csv_data_dict["data"], \
        processed_time_list, \
        os, \
        browser, \
        dl.csv_data_dict["target"]])

    bl = BalanceDataset(res_np, {"target" : -1}, "not classification").run()
    """
    return bl


if __name__ == "__main__":
    features("data4", "0.5_10_1.csv")

