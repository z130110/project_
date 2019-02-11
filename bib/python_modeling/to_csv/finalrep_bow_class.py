from etl.transformations.process_time import ProcessTime
from etl.transformations.getavg_time import GetAvgTime
from etl.transformations.balance_dataset import BalanceDataset
from etl.transformations.login_time import LoginTime
from etl.transformations.page_encode import PageEncode
from etl.transformations.normalize_time import NormalizeTime
from etl.transformations.process_useragents import ProcessUserAgents
from etl.load.data_loader import DataLoader
from sklearn.model_selection import train_test_split
from etl.transformations.process_time import ProcessTime
from etl.transformations.time_discretization import TimeDiscretize
from etl.transformations.bow import BOW
from etl.transformations.balance_dataset_class import BalanceDatasetClass


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections

def split_and_balance_data(np_arr, balance):
    train, test = train_test_split(np_arr, test_size=0.05, random_state=42)
    if balance:
        train_balanced = BalanceDatasetClass(train, {"target" : -1}, "not classification").run()
        return train_balanced, test
    return train, test

def save_csv_to_np(data_dir, file_name, percent_to_read = 1):
    features(data_dir, file_name, percent_to_read)

def t2(loaded_data):
    #print("processing t1")
    save_dir = "/data/numpy_conversions/data10/"
    
    avg_time, avg_std = GetAvgTime(loaded_data.csv_data_dict["combined_eventtimestamp"]).run()
    avg_time = avg_time.tolist()
    avg_std = avg_std.tolist()
    
    login1,login2,login3,login4 = LoginTime(loaded_data.csv_data_dict["sessionstarttime_hour"],\
             loaded_data.csv_data_dict["sessionstarttime_minute"],loaded_data.csv_data_dict["sessionstarttime_weekday"]).run()
    
    os, browser = ProcessUserAgents(loaded_data.csv_data_dict["os"], \
        loaded_data.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    
    bow = BOW(loaded_data.csv_data_dict["combined_pagelocation"]).run()
    bow = bow.tolist()

    res_np = loaded_data.to_numpy([bow, \
        os, \
        browser, \
        login1, \
        login2, \
        login3, \
        login4, \
        avg_time, \
        avg_std, \
        loaded_data.csv_data_dict["target"]])
    balanced = False
    train, test = split_and_balance_data(res_np,False)
    if balanced:
        filename = save_dir + "t2classbow_"
    else:
        filename = save_dir + "t2classbowunbalanced_"
    #filename = save_dir + "t2classbowunbalanced_"
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
    t2(dl)

if __name__ == "__main__":
    save_csv_to_np("data10", "1.0_10_3.csv", 0.1)
    #save_csv_to_np("data10", "test.csv")
    
    #save_csv_to_np("data9", "test.csv")
