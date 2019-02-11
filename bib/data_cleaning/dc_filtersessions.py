from pyspark.sql.window import Window
import pyspark.sql.functions as func
from pyspark.sql import Row
from pyspark.sql.functions import col

class Filtersessions(object):
    """
    Filter session based on number of clicks for each session, and balance dataset if requested
    """
    def __init__(self, cp_table, target_table, min_clicks,balanced,nm_rows):
        self.filtered_targets   =\
            self.targets_for_n_clicks(target_table,cp_table,min_clicks)
        if balanced:
            self.filtered_targets =self.balance_dataset(self.filtered_targets, "target", n=nm_rows)

    def sessions_with_more_than_n_clicks(self,click_table, n=5):
        """
        Return sessionnumbers, where the number of clicks >= n
        @param click_table : The click table 
        @param n           : Number of clicks, to include 
        @return sessionnumbers where the number of clicks for each session >= n
        """
        click_counts_sessionnumber = click_table \
            .groupBy(click_table.sessionnumber) \
            .agg(func.count("*").alias("click_counts"))

        sessionnumber_with_n_clicks = click_counts_sessionnumber \
            .select(click_counts_sessionnumber.sessionnumber) \
            .where((click_counts_sessionnumber.click_counts >= n) & (click_counts_sessionnumber.click_counts <= 50))

        return sessionnumber_with_n_clicks
    
    def targets_for_n_clicks(self,target_df, click_table, n_clicks=5):
        """
        @param target_df   : input target dataframe
        @param click_table : input click table dataframe
        @parm  n_clicks    : only retrieve sessions higher than n number of clicks
        @return targets for sessions with more than n number of clicks
        """
        filtered_sessions = self.sessions_with_more_than_n_clicks(click_table,n_clicks)
        return filtered_sessions\
            .join(target_df, filtered_sessions.sessionnumber ==\
                             target_df.sessionnumber, 'inner')\
            .select(filtered_sessions.sessionnumber, target_df.target)
            
        #.select(filtered_sessions.sessionnumber, target_df.target,\
        #            target_df.aftlnr)


    def balance_dataset(self,target_df, column_name, n):
        """ 
        Return samples of rows. Each distinct column_name will have n rows.
        @param target_df   : Input target data frame
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
            .select(target_df["sessionnumber"], target_df[column_name], target_df.aftlnr)

        w = Window.partitionBy(func.col(column_name)).orderBy(func.col(column_name))

        sampled = targets_20000 \
            .withColumn("rn_", func.row_number().over(w))  \
            .where(func.col("rn_") <= n) \
            .drop("rn_")
        return sampled
    
    def get_filtered(self):
        return self.filtered_targets