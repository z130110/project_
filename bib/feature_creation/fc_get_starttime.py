from pyspark.sql.functions import col, expr, when, lower, collect_set,collect_list,udf
import datetime
import pytz

def utc_to_time(eventtimestamp):
    timezone="Europe/Amsterdam"
    eventtimestamp_seconds = int(eventtimestamp) / 1000.
    datetime_format = datetime.datetime.fromtimestamp(eventtimestamp_seconds)
    return datetime_format.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(timezone))

def get_weekday(eventtimestamp):
    eventtimestamp_seconds = int(eventtimestamp) / 1000.
    datetime_format = datetime.datetime.fromtimestamp(eventtimestamp_seconds)
    weekday = datetime_format.weekday()
    return weekday

def get_hour(eventtimestamp):
    eventtimestamp_seconds = int(eventtimestamp) / 1000.
    datetime_format = datetime.datetime.fromtimestamp(eventtimestamp_seconds)
    hour = datetime_format.hour
    return hour

def get_minute(eventtimestamp):
    eventtimestamp_seconds = int(eventtimestamp) / 1000.
    datetime_format = datetime.datetime.fromtimestamp(eventtimestamp_seconds)
    minute = datetime_format.minute
    return minute

class GetStartTime(object):
    def __init__(self, input_df):
        df_with_sessionsstart = self.collect_start_time(input_df)
        self.df_with_dates = self.add_day_time(df_with_sessionsstart)
        
    def collect_start_time(self,df):
        get_first_udf = udf(lambda x : x[0])
        return df.withColumn("sessionstarttime",get_first_udf("combined_eventtimestamp"))#.drop("combined_eventtimestamp")
        
        """
        return df.groupBy("sessionnumber").agg(collect_list("eventtimestamp") \
                                          .alias("combined_time")) \
                                          .withColumn("sessionstarttime",get_first_udf("combined_time")) \
                                          .drop("combined_time")
        """
        
    def add_day_time(self,df):
        get_weekday_udf = udf(get_weekday)
        get_hour_udf = udf(get_hour)
        get_minute_udf = udf(get_minute)
        df = df.withColumn("sessionstarttime_weekday", get_weekday_udf(df['sessionstarttime']))
        df = df.withColumn("sessionstarttime_hour", get_hour_udf(df['sessionstarttime']))
        df = df.withColumn("sessionstarttime_minute", get_minute_udf(df['sessionstarttime']))
        #df.show(5)
        return df
    
    def run(self):
        return self.df_with_dates