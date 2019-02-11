
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
		raw_data_pipe = DSTest(self.spark)
		raw_data_df = raw_data_pipe.run()

		return raw_data_df
		#filtered_data_pipe = DSRawDataExtraction(self.spark \
		#, raw_data_df)

class DSTest:
	def __init__(self \
		, spark_session \
		, database_name = "34np_project_celebrus_ku" \
		, visitor_name = "hd_wt_visitor" \
		, ewwh_ebank_agree_h_name = "34np_gard_ewwh_ebank_agree_h" \
		, abt_db_p_name = "demoss_gendab_abt_sync_abt_db_p"):
		self.spark = spark_session
		self.visitor_table = self.spark.table(database_name + "." + visitor_name)
		self.ewwh_ebank_agree_h_table = self.spark.table(database_name + "." + ewwh_ebank_agree_h_name)
		self.abt_db_p_table = self.spark.table(database_name + "." + abt_db_p_name)

	def run(self):
		final_targets = self.raw_data_filtered()
		return final_targets

	def raw_data_filtered(self):
		"""
		Joins ewwh, abt, and visitor, while filtering out duplicates and
		sorting by date.
		"""

		# ewwh_ebank_agree_h_table contains [knid, ebanking-agreement, ..]
		# As there are multiple rows with identical aft_hav_ip_id(knid), choose the first.
		# NOTE: groupBy is order-preserving relative to functions (eg. first)
		distinct_knid_ebank_agree = self.ewwh_ebank_agree_h_table \
			.select(self.ewwh_ebank_agree_h_table.aft_hav_ip_id.substr(1, 10).alias('knid') \
				, self.ewwh_ebank_agree_h_table.aftlnr) \
			.distinct()

		"""
			.orderBy(self.ewwh_ebank_agree_h_table.aft_hav_ip_id \
					, self.ewwh_ebank_agree_h_table.mtts) \
			.groupBy(self.ewwh_ebank_agree_h_table.aft_hav_ip_id.substr(1, 10).alias("knid")) \
			.agg(func.first(self.ewwh_ebank_agree_h_table.aftlnr).alias("aftlnr"))
		"""
		#print("dis", distinct_knid_ebank_agree.count())
		#print("dis",distinct_knid_ebank_agree.count())

		# self.abt_db_p_table contains [knid, age, ..]
		knid_age_unique = self.abt_db_p_table \
			.groupBy(self.abt_db_p_table.knid) \
			.agg(func.max(self.abt_db_p_table.customer_age).alias("customer_age"))
		#print("knid_age_unique", knid_age_unique.count())
		#print("knid_age_unique", knid_age_unique.count())

		# Join the two tables. knid_agree_age = [knid, age, ebanking-agreement]
		knid_agree_age = knid_age_unique \
			.join(distinct_knid_ebank_agree \
				, distinct_knid_ebank_agree.knid == knid_age_unique.knid \
				, "inner") \
			.select(knid_age_unique.knid \
				, distinct_knid_ebank_agree.aftlnr \
				, knid_age_unique.customer_age)
		#print("knid agree", knid_agree_age.count())
		#print("knid agree",knid_agree_age.count())

		# Joined with the visitor table. 

		final_targets = knid_agree_age.join(self.visitor_table \
											, self.visitor_table.profileuiid == knid_agree_age.aftlnr \
											, "left_semi")
		#print("final",final_targets.count())
		#print("final",final_targets.count())

		visitor_table_unique_target = self.visitor_table \
			.select(self.visitor_table.sessionnumber \
					, func.upper(self.visitor_table.profileuiid).alias("profileuiid")) \
			.distinct()
		#print("visitor",visitor_table_unique_target.count())
		#print("visitor",visitor_table_unique_target.count())

		final_targets = visitor_table_unique_target \
			.join(final_targets \
				, visitor_table_unique_target.profileuiid == final_targets.aftlnr \
				, 'inner')
		#print("final2",final_targets.count())
		#print("final2",final_targets.count())

		final_targets_corrected = final_targets \
			.select(final_targets.sessionnumber \
					, final_targets.profileuiid.alias("aftlnr")
					, final_targets.knid
					, final_targets.customer_age.alias("target"))
		#print("final3",final_targets_corrected.count())
		#print("final3",final_targets_corrected.count())

		return final_targets_corrected
