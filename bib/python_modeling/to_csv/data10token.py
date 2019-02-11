import sys
from etl.transformations.tokenize_pages import TokenizePages
from etl.load.data_loader import DataLoader
from etl.transformations.process_time import ProcessTime
from etl.transformations.getavg_time import GetAvgTime
from etl.transformations.balance_dataset import BalanceDataset
from etl.transformations.login_time import LoginTime
from etl.transformations.page_encode import PageEncode
from etl.transformations.normalize_time import NormalizeTime
from etl.transformations.process_useragents import ProcessUserAgents

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

def save_csv_to_np(data_dir, file_name, percent_to_read = 0.9):
    #save_dir = "/data/numpy_conversions/data9/" + file_name[:-4] + "_" + "tokenized"
    features(data_dir, file_name, percent_to_read)
    #print("HER")
    #print(dl.shape)
    #np.save(save_dir, dl)

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
    processed_time = NormalizeTime(dl.csv_data_dict["combined_eventtimestamp"]).run()
    
    tokens = TokenizePages(dl.csv_data_dict["combined_pagelocation"]).run()
    login1, login2 = LoginTime(dl.csv_data_dict["sessionstarttime_hour"], dl.csv_data_dict["sessionstarttime_minute"]).run()
    
    os, browser = ProcessUserAgents(dl.csv_data_dict["os"], \
        dl.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    res_np = dl.to_numpy([
        tokens, \
        processed_time, \
        os, \
        browser, \
        login1, \
        login2, \
        dl.csv_data_dict["target"]])
    
    train, test = split_and_balance_data(res_np)
    save_dir = "/data/numpy_conversions/data10/"
    np.save(save_dir + "tokenized_train.npy", train)
    np.save(save_dir + "tokenized_test.npy", test)
    
if __name__ == "__main__":
    save_csv_to_np("data10", "0.8_15_2.csv")
    #save_csv_to_np("data10", "test.csv")