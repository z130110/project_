import sys
sys.path.append("/home/Workdir/bib/constants")
import databases
from dc_filtersessions import Filtersessions
from dc_cleanpagelocation import CleanPageLocation
from dc_useragents import Useragents
from dc_filter_pages import FilterPages
from dc_get_starttime import GetStartTime
from dc_filter_adjacent_clicks import FilterAdjacentClicks
import pyspark.sql.functions as func
from pyspark.sql.functions import col, expr, when, lower, collect_set,collect_list

class DC(object):
    """
    Datacleaner component of the pipeline
    Takes as input a dataframe, containing sessionnumber, aftlnr, kNid, target
    @param spark        : spark session
    @param target_df    : dataframe holding the target value, output from data selection step
    @param min_clicks   : min number of clicks allowed for each session
    @param balanced_set : whether or not the returned data_set should be balanced
    """
    def __init__(self, spark, target_df, min_clicks, balanced_set=False, balanced_rows=20000):
        self.spark         = spark
        self.target_df     = target_df
        self.click_table   = spark.table(databases.table_click)
        self.page_table    = spark.table(databases.table_page)
        self.useragent_df  = Useragents(self.spark).run()
        self.min_clicks    = min_clicks
        self.balanced      = balanced_set
        self.balanced_rows = balanced_rows
        self.cleaned_data  = self.preprocess_tables(self.click_table,self.page_table)
 
    def preprocess_tables(self,click_df,page_df):
        """
        Performs the datacleaning pipeline
        @param click_df  : the click table dataframe
        @param page_df   : the page table dataframe
        @param target:df : the dataframe returned from data selection step
        @return : returns a filtered dataframe
        """
        cp_df                = self.take_only_page(page_df)
        cp_df                = self.add_all_time_stamps(cp_df)
        cleaned_cp           = CleanPageLocation(cp_df).get_filtered()
        filtered_df          = FilterPages(cleaned_cp).run()
        filtered_adjacent_df = FilterAdjacentClicks(filtered_df).run()
        filtered_target      = Filtersessions(filtered_adjacent_df, self.target_df, self.min_clicks,\
                                 self.balanced, self.balanced_rows).get_filtered()
        filtered_cp_target   = self.join_cp_target(filtered_adjacent_df,filtered_target,self.useragent_df)
        return filtered_cp_target.orderBy(["sessionnumber","eventtimestamp"],ascending=[1,1]).distinct()
    
    def take_only_page(self,page_df):
        return page_df.select("sessionnumber","pagelocation","eventtimestamp","pageinstanceid")\
                      .orderBy(["sessionnumber","eventtimestamp"],ascending=[1,1])
        
    def add_all_time_stamps(self,cp_df):
        df_collected = cp_df.groupBy("sessionnumber")\
            .agg(collect_list("eventtimestamp").alias("all_eventtimestamp"))
        df_joined = cp_df.join(df_collected, "sessionnumber", "inner")
        return df_joined
    
    def join_page_click(self,click_df,page_df):
        """
        Joins page and click table
        @param click_df : click table
        @param page_df  : page table
        returns the joined click and page table
        """
        click_df = click_df.orderBy(['sessionnumber','eventtimestamp'],ascending=[1,1])
        page_df  = page_df.orderBy(['sessionnumber'],ascending=[1])
                                    
        return click_df.join(page_df, "pageinstanceid", "inner")\
             .orderBy([click_df.sessionnumber,click_df.eventtimestamp],ascending=[1,1])\
             .select(click_df.sessionnumber, "objectalttext", "objectsourcetype", "objecthref", "pagelocation")
                       
    def join_cp_target(self,cp_df,target_df,useragent_df):
        """
        Join target and page-click table 
        """
        target_user_agent = target_df.join(useragent_df, target_df.sessionnumber == useragent_df.sessionnumber,"inner")\
            .select(target_df.sessionnumber, target_df.target,useragent_df.browser,\
             useragent_df.os)
        #target_df.aftlnr
        return cp_df.join(target_user_agent, target_user_agent.sessionnumber == cp_df.sessionnumber,"inner")\
                    .select(cp_df.sessionnumber, "pagelocation","target","browser","os",\
                  "eventtimestamp","pageinstanceid","all_eventtimestamp")
        
    def run(self):
        return self.cleaned_data