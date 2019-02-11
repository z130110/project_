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
from etl.transformations.convert_class import ConvertClass


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import collections

def split_and_balance_data(np_arr, balance):
    train, test = train_test_split(np_arr, test_size=0.05, random_state=42)
    if balance:
        train_balanced = BalanceDataset(train, {"target" : -1}).run()
        return train_balanced, test
    return train, test

def split_and_balance_data_class(np_arr, balance):
    train, test = train_test_split(np_arr, test_size=0.1, random_state=42)
    if balance:
        train_balanced = BalanceDatasetClass(train, {"target" : -1}).run()
        return train_balanced, test
    return train, test


def save_csv_to_np(data_dir, file_name, percent_to_read = 1):
    features(data_dir, file_name, percent_to_read)
    #print(dl.shape)
    #np.save(save_dir, dl)

def tword2vec(loaded_data):
    #print("processing t1")
    save_dir = "/data/numpy_conversions/finalreptask3/"
    loaded_data.csv_data_dict["all_eventtimestamp"]
    avg_time, avg_std = GetAvgTime(loaded_data.csv_data_dict["all_eventtimestamp"]).run()
    avg_time = avg_time.tolist()
    avg_std = avg_std.tolist()
    event_time = NormalizeTime(loaded_data.csv_data_dict["all_eventtimestamp"],cut_off = 10).run()
    
    login1,login2,login3,login4 = LoginTime(loaded_data.csv_data_dict["sessionstarttime_hour"],\
             loaded_data.csv_data_dict["sessionstarttime_minute"],loaded_data.csv_data_dict["sessionstarttime_weekday"]).run()
    
    os, browser = ProcessUserAgents(loaded_data.csv_data_dict["os"], \
        loaded_data.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()

    reg_np = loaded_data.to_numpy([loaded_data.csv_data_dict["data"], \
        event_time, \
        os, \
        browser, \
        login1, \
        login2, \
        login3, \
        login4, \
        avg_time, \
        avg_std, \
        loaded_data.csv_data_dict["target"]])
    
    #train, test = split_and_balance_data(reg_np,False)
    #filename = save_dir + "t3_word2vec_unbalanced_"
    #np.save(filename + "train.npy", train)
    #np.save(filename + "test.npy", test)
    
    train, test = split_and_balance_data_class(reg_np,True)
    filename = save_dir + "t3_word2vec_balanced_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    
    
    train, test = split_and_balance_data_class(reg_np,False)
    filename = save_dir + "t3_word2vec_unbalanced_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    """
    train, test = split_and_balance_data_class(reg_np,True)
    filename = save_dir + "t2_word2vec_class_balanced_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    """

def cut_page(session):
    out = []
    for i, click in enumerate(session):
        if "danskebank.dk/privat/bliv-kunde" in click:
        #if "frontpage" in click:
            print("returning")
            return out
        else:
            out.append(click)
    return out
def cut_pages_for_sessions(clicks):
    fin_arr = []
    for session in clicks:
        fin_arr.append(cut_page(session))
    return fin_arr
    
def tbow(loaded_data):
    save_dir = "/data/numpy_conversions/finalreptask3/"
    
    avg_time, avg_std = GetAvgTime(loaded_data.csv_data_dict["all_eventtimestamp"]).run()
    avg_time = avg_time.tolist()
    avg_std = avg_std.tolist()
    event_time = NormalizeTime(loaded_data.csv_data_dict["all_eventtimestamp"],cut_off = 10).run()

    
    login1,login2,login3,login4 = LoginTime(loaded_data.csv_data_dict["sessionstarttime_hour"],\
             loaded_data.csv_data_dict["sessionstarttime_minute"],loaded_data.csv_data_dict["sessionstarttime_weekday"]).run()
    
    os, browser = ProcessUserAgents(loaded_data.csv_data_dict["os"], \
        loaded_data.csv_data_dict["browser"]).run()
    os = os.tolist()
    browser = browser.tolist()
    
    
    clicks = loaded_data.csv_data_dict["combined_pagelocation2"]
    bow = BOW(clicks,json_name="task3_new").run()
    #bow = bow.tolist()
    
    reg_np = loaded_data.to_numpy([bow, \
        event_time, \
        os, \
        browser, \
        login1, \
        login2, \
        login3, \
        login4, \
        avg_time, \
        avg_std, \
        loaded_data.csv_data_dict["target"]])

    train, test d= split_and_balance_data_class(reg_np,False)
    filename = save_dir + "t3_bow_unbalanced_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    
    train, test = split_and_balance_data_class(reg_np,True)
    filename = save_dir + "t3_bow_balanced_"
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)

    """
    class_np = loaded_data.to_numpy([bow, \
        os, \
        browser, \
        login1, \
        login2, \
        login3, \
        login4, \
        avg_time, \
        avg_std, \
        ConvertClass(loaded_data.csv_data_dict["target"]).run()])
    
    """
    """
    train, test = split_and_balance_data_class(class_np,False)
    filename = save_dir + "t2_bow_class_unbalanced_"
    
    np.save(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    """
    """
    train, test = split_and_balance_data_class(class_np,True)
    filename = save_dir + "t2_bow_class_balanced_"
    np(filename + "train.npy", train)
    np.save(filename + "test.npy", test)
    """
def features(data_dir, file_name, percent_to_read):
    dl = DataLoader(data_dir, file_name, features_to_use= \
        ["sessionstarttime_weekday", \
        "sessionstarttime_hour", \
         "sessionstarttime_minute", \
        "data", \
        "os", \
        "browser", \
        "combined_pagelocation2", \
        "combined_eventtimestamp", \
        "target",\
        "all_eventtimestamp"],percent_to_read=percent_to_read)
    tbow(dl)
    #tword2vec(dl)
    
if __name__ == "__main__":
    save_csv_to_np("task3", "1.0_10_3.csv",1.0)
    #save_csv_to_np("data10", "test1.csv")
    
    #save_csv_to_np("data9", "test.csv")
