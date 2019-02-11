from etl.transformations.process_time import ProcessTime
from etl.transformations.getavg_time import GetAvgTime
from etl.transformations.balance_dataset import BalanceDataset
from etl.transformations.login_time import LoginTime
from etl.transformations.page_encode import PageEncode
from etl.transformations.normalize_time import NormalizeTime
from etl.transformations.process_useragents import ProcessUserAgents
from etl.load.data_loader import DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections

def save_csv_to_np(data_dir, file_name, percent_to_read = 0.7):
    save_dir = "/data/numpy_conversions/data9/" + file_name[:-4] + "_" + "balanced_2_with_login_time"
    dl = features(data_dir, file_name, percent_to_read)
    #print("HER")
    print(dl.shape)
    np.save(save_dir, dl)

def features(data_dir, file_name, percent_to_read):
    dl = DataLoader(data_dir, file_name, features_to_use= \
        ["sessionstarttime_weekday", \
        "sessionstarttime_hour", \
         "sessionstarttime_minute", \
        "data", \
        "os", \
        "browser", \
        "combined_pagelocation", \
        "combined_eventtimestamp", \
        "target"],percent_to_read=percent_to_read)

    page_encoded_data = PageEncode(dl.csv_data_dict["combined_pagelocation"]).run()
    #print(page_encoded_data)
    #print(list(map(len, page_encoded_data)))

    event_time = NormalizeTime(dl.csv_data_dict["combined_eventtimestamp"]).run()
    #print(event_time)

    login1, login2 = LoginTime(dl.csv_data_dict["sessionstarttime_hour"], dl.csv_data_dict["sessionstarttime_minute"]).run()

    os, browser = ProcessUserAgents(dl.csv_data_dict["os"], \
        dl.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()

    res_np = dl.to_numpy([page_encoded_data, \
        event_time, \
        os, \
        browser, \
        login1, \
        login2, \
        dl.csv_data_dict["target"]])

    b1 = BalanceDataset(res_np, {"target" : -1}, "not classification").run()
    return res_np


if __name__ == "__main__":
    save_csv_to_np("data9", "0.5_10_1.csv")
