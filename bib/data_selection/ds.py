import numpy as np
from matplotlib import pyplot as plt
import pyspark.sql.functions as func
from pyspark.sql import Row
import pandas as pd
from pyspark.sql.window import Window
from random import randint,seed
from pyspark.sql.functions import col, expr, when, lower, collect_set,collect_list, abs
from pyspark.sql.functions import concat, col, lit, regexp_replace,udf,concat_ws,rand,randn
from pyspark.sql.types import DoubleType,IntegerType, ArrayType
import os

ABS_PATH=os.path.abspath(os.path.dirname(__file__))
#from ds_filter_raw_data import DSFilterData
from ds_raw_data_extraction import DSRawDataExtraction


class DS:
    def __init__(self \
        , spark \
        , database_name = "34np_project_celebrus_ku" \
        , visitor_name = "hd_wt_visitor" \
        , ewwh_ebank_agree_h_name = "34np_gard_ewwh_ebank_agree_h" \
        , abt_db_p_name = "demoss_gendab_abt_sync_abt_db_p"
        , click_name = "hd_wt_click" \
        , column_name = "target" \
        , nr_rows = 20000):

        self.spark = spark
        self.visitor_table = spark.table(database_name + "." + visitor_name)
        self.ewwh_ebank_agree_h_table = spark.table(database_name + "." + ewwh_ebank_agree_h_name)
        self.click_table = spark.table(database_name + "." + click_name)
        self.abt_db_p_table = spark.table(database_name + "." + abt_db_p_name)
        self.n = nr_rows

        self.visitor_table = self.spark.table(database_name + "." + visitor_name)
        self.ewwh_ebank_agree_h_table = self.spark.table(database_name + "." + ewwh_ebank_agree_h_name)
        self.abt_db_p_table = self.spark.table(database_name + "." + abt_db_p_name)

    def run(self):
        raw_data_pipe = DSRawDataExtraction(self.spark)
        raw_data_df = raw_data_pipe.run()

        return raw_data_df
        #filtered_data_pipe = DSRawDataExtraction(self.spark \
        #, raw_data_df)
