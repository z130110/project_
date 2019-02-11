from etl.transformations.process_time import ProcessTime
from etl.transformations.getavg_time import GetAvgTime
from etl.transformations.balance_dataset import BalanceDataset
from etl.transformations.login_time import LoginTime
from etl.transformations.page_encode import PageEncode
from etl.transformations.normalize_time import NormalizeTime
from etl.transformations.process_useragents import ProcessUserAgents
from etl.load.data_loader import DataLoader
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections

def split_and_balance_data(np_arr):
    train, test = train_test_split(np_arr, test_size=0.20, random_state=42)
    train_balanced = BalanceDataset(train, {"target" : -1}, "not classification").run()
    return train_balanced, test

def save_csv_to_np(data_dir, file_name, percent_to_read = 1):
    save_dir = "/data/numpy_conversions/data9/" + file_name[:-4] + "_" + "balanced_2_with_login_time"
    features(data_dir, file_name, percent_to_read)
    #print(dl.shape)
    #np.save(save_dir, dl)

def t1(loaded_data):
    print("processing t1")
    save_dir = "/data/numpy_conversions/data9/"
    page_encoded_data = PageEncode(loaded_data.csv_data_dict["combined_pagelocation"]).run()
    event_time = NormalizeTime(loaded_data.csv_data_dict["combined_eventtimestamp"]).run()
    
    login1, login2 = LoginTime(loaded_data.csv_data_dict["sessionstarttime_hour"],\
                               loaded_data.csv_data_dict["sessionstarttime_minute"]).run()

    os, browser = ProcessUserAgents(loaded_data.csv_data_dict["os"], \
        loaded_data.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()

    res_np = loaded_data.to_numpy([page_encoded_data, \
        event_time, \
        os, \
        browser, \
        login1, \
        login2, \
        loaded_data.csv_data_dict["target"]])
    train, test = split_and_balance_data(res_np)
    filename = save_dir + "t1_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)

    
def t2(loaded_data):
    print("processing t2")
    save_dir = "/data/numpy_conversions/data9/"
    processed_time = GetAvgTime(loaded_data.csv_data_dict["combined_eventtimestamp"]).run()
        
    processed_time_list = processed_time.tolist()
    
    login1, login2 = LoginTime(loaded_data.csv_data_dict["sessionstarttime_hour"],\
                               loaded_data.csv_data_dict["sessionstarttime_minute"]).run()

    os, browser = ProcessUserAgents(loaded_data.csv_data_dict["os"], \
        loaded_data.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    t = time.time()
    res_np = loaded_data.to_numpy([loaded_data.csv_data_dict["data"], \
        processed_time_list, \
        os, \
        browser, \
        login1, \
        login2, \
        loaded_data.csv_data_dict["target"]])
    
    train, test = split_and_balance_data(res_np)
    filename = save_dir + "t2_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    
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
    #t1(dl)
    t2(dl)

if __name__ == "__main__":
    save_csv_to_np("data9", "0.5_10_3.csv")
    #save_csv_to_np("data9", "test.csv")
