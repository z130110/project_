import sys
sys.path.insert(0, "/home/Workdir/bib/python_modeling/etl/transformations")
sys.path.insert(0, "/home/Workdir/bib/python_modeling/etl/load")
from data_loader import DataLoader
from process_useragents import ProcessUserAgents
from process_time import ProcessTime
from getavg_time import GetAvgTime
from balance_dataset_class import BalanceDatasetClass
from login_time import LoginTime
from convert_class import ConvertClass

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections

def save_csv_to_np(data_dir, file_name, percent_to_read =1.0):
    save_dir = "/data/numpy_conversions/data9class/" + file_name[:-4] + "_" + "with_login_time"
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

    t = time.time()
    processed_time = GetAvgTime(dl.csv_data_dict["combined_eventtimestamp"]).run()
    
    #processed_time = ProcessTime(dl.csv_data_dict["combined_pagelocation"], \
    #    dl.csv_data_dict["combined_eventtimestamp"]).run()
    processed_time_list = processed_time.tolist()
    t = time.time()
    
    login1, login2 = LoginTime(dl.csv_data_dict["sessionstarttime_hour"], dl.csv_data_dict["sessionstarttime_minute"]).run()

    os, browser = ProcessUserAgents(dl.csv_data_dict["os"], \
        dl.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    t = time.time()

    targets_class = ConvertClass(dl.csv_data_dict["target"]).run()
    
    res_np = dl.to_numpy([dl.csv_data_dict["data"], \
        processed_time_list, \
        os, \
        browser, \
        login1, \
        login2, \
        targets_class])
    balanced = BalanceDatasetClass(res_np, {"target" : -1}, "not classification").run()
    #print(balanced[:,-1])
    return balanced

if __name__ == "__main__":
    #save_csv_to_np("data9", "test.csv")
    save_csv_to_np("data9", "0.5_10_1.csv")
    save_csv_to_np("data9", "0.5_10_2.csv")
    save_csv_to_np("data9", "0.9_10_3.csv")
    """
    save_csv_to_np("data8", "0.9_10_4.csv")
    save_csv_to_np("data8", "0.9_10_5.csv")
    save_csv_to_np("data8", "0.9_10_6.csv")
    save_csv_to_np("data8", "0.9_10_7.csv")
    save_csv_to_np("data8", "0.9_10_8.csv")
    save_csv_to_np("data8", "0.9_10_9.csv")
    save_csv_to_np("data8", "0.9_10_10.csv")
    save_csv_to_np("data8", "0.9_15_1.csv")
    save_csv_to_np("data8", "0.9_15_2.csv")
    save_csv_to_np("data8", "0.9_15_3.csv")
    save_csv_to_np("data8", "0.9_15_4.csv")
    save_csv_to_np("data8", "0.9_15_5.csv")
    save_csv_to_np("data8", "0.9_15_6.csv")
    save_csv_to_np("data8", "0.9_15_7.csv")
    save_csv_to_np("data8", "0.9_15_8.csv")
    save_csv_to_np("data8", "0.9_15_9.csv")
    save_csv_to_np("data8", "0.9_15_10.csv")
    save_csv_to_np("data8", "0.9_35_1.csv")
    save_csv_to_np("data8", "0.9_35_2.csv")
    save_csv_to_np("data8", "0.9_35_3.csv")
    save_csv_to_np("data8", "0.9_35_4.csv")
    save_csv_to_np("data8", "0.9_35_5.csv")
    save_csv_to_np("data8", "0.9_35_6.csv")
    save_csv_to_np("data8", "0.9_35_7.csv")
    save_csv_to_np("data8", "0.9_35_8.csv")
    save_csv_to_np("data8", "0.9_35_9.csv")
    save_csv_to_np("data8", "0.9_35_10.csv")
    """