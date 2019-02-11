import sys
#sys.path.insert(0, "/home/Workdir/bib/python_modeling/etl/extract")
from etl.extract.read_csv import CsvReader
import numpy as np

class DataLoader():
    def __init__(self, data_dir, file_name, features_to_use, percent_to_read=1.0, counts_for_each_target=sys.maxsize):
        self.features_to_use = features_to_use
        self.csv_data_dict = CsvReader(data_dir, file_name, features_to_read=features_to_use, percent_to_read=percent_to_read, counts_for_each_target=counts_for_each_target).read_raw_csv()

    def to_row_column_format(self, column_list):
        nr_rows = len(column_list[0])
        res = []
        # From column-row to row-column
        raw_rows = [[row[i] for row in column_list] for i in range(len(column_list[0]))]

        for raw_row in raw_rows:
            row = []
            for data_item in raw_row:
                if type(data_item) != list:
                    row.append([data_item])
                else:
                    row.append(data_item)
            res.append(list(row))
        return res

    def to_numpy(self, column_list):
        res = []
        column_data = self.to_row_column_format(column_list)
        for row in column_data:
            res.append([float(data_item) for data_item in self.flat_list(row)])
        return np.array(res)

    def flat_list(self, x):
        res = []
        for row in x:
            if type(row[0]) is np.ndarray:
                row = list(row[0])
            for item in row:
                res.append(item)
        return res

if __name__ == "__main__":
    print("Comment out below for test")
    """
    dl = DataLoader("data4", "0.5_10_1.csv", features_to_use= \
        ["sessionstarttime", \
        "sessionstarttime_weekday", \
        "combined_pagelocation",
        "combined_eventtimestamp"])
    res = dl.to_row_column_format()
    """
