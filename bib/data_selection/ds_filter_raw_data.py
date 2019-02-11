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

class DSFilterData:
	def __init__(self \
		, spark \
		, target_df \
		, database_name = "34np_project_celebrus_ku" \
		, click_name = "hd_wt_click" \
		, column_name = "target" \
		, nr_rows = 20000):
		self.spark = spark
		self.target_df = target_df
		self.click_table = spark.table(database_name + "." + click_name)
		self.column_name = column_name
		self.n = nr_rows

	def run(self):
		#final_targets = self.balance_dataset(self.target_df \
		#	, self.column_name \
		#	, self.n)
		final_targets = self.targets_for_n_clicks(self.target_df, self.click_table)
		return final_targets

	def targets_for_n_clicks(self, target_df, click_table, n_clicks=5, n_min_number_rows=20000):
		sessionnumbers_5_clicks = self.sessions_with_more_than_n_clicks(click_table, n=n_clicks)
		session_5_clics_joined_with_target = sessionnumbers_5_clicks \
			.join(target_df \
				, sessionnumbers_5_clicks.sessionnumber == target_df.sessionnumber \
				, 'inner') \
			.select(sessionnumbers_5_clicks.sessionnumber \
				, target_df.knid \
				, target_df.aftlnr \
				, target_df.target)

		#res = self.balance_dataset(session_5_clics_joined_with_target \
		#		, column_name="target" \
		#		, n=n_min_number_rows)
		return session_5_clics_joined_with_target

	"""
	def balance_dataset(self, target_df, column_name = "target", n = 20000):
		"""
		Return samples of rows. Each distinct column_name will have n rows.
		@param target_df   : Input target - should be output from previous selection pipe
		@param column_name : Name of column to group by.
		@param n           : The number of rows per column_name
		@return output dataframe with output column
		"""
		#TODO: not all sessionnumbers distinct: 1360000 total vs. 1350894 distinct

		counts_of_age = target_df \
			.groupBy(func.col(column_name)) \
			.agg(func.count("*").alias("aggregated_count")) \

		counts_of_age = counts_of_age \
			.orderBy(func.col(column_name))

		ages_to_include = counts_of_age.where(counts_of_age.aggregated_count >= n)

		targets_20000 = target_df \
			.join(ages_to_include \
				, target_df[column_name] == ages_to_include[column_name] \
				, 'inner') \
			.select(target_df.sessionnumber, target_df[column_name], target_df.aftlnr)

		w = Window.partitionBy(func.col(column_name)).orderBy(func.col(column_name))

		sampled = targets_20000 \
			.withColumn("rn_", func.row_number().over(w))  \
			.where(func.col("rn_") <= n) \
			.drop("rn_")

		return sampled

	def sessions_with_more_than_n_clicks(self, click_table, n=5):
		"""
		Return Return sessionnumbers, where the number of clicks >= n
		@param click_table : The click table
		@param n           : Number of clicks, to include
		@return sessionnumbers where the number of clicks for each session >= n
		"""
		click_counts_sessionnumber = click_table \
			.groupBy(click_table.sessionnumber) \
			.agg(func.count("*").alias("click_counts"))

		sessionnumber_with_n_clicks = click_counts_sessionnumber \
			.select(click_counts_sessionnumber.sessionnumber) \
			.where(click_counts_sessionnumber.click_counts >= n)

		return sessionnumber_with_n_clicks
	"""
