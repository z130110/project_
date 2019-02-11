import sys
sys.path.append("/home/Workdir/bib/constants")
import databases
from pyspark.sql.functions import col, regexp_replace, lower, from_unixtime, from_utc_timestamp, date_format,udf
from pyspark.sql.functions import *
import datetime
import pytz

def map_browser_name(browser):
    if browser is None:
        return "Other"
    browser = browser.lower()
    if "edge" in browser:
        return "Edge"
    elif "chrome" in browser:
        return "Chrome"
    elif "explorer" in browser:
        return "Explorer"
    elif "mozilla" in browser:
        return "Mozilla"
    elif "opera" in browser:
        return "Opera"
    elif "safari" in browser:
        return "Safari"
    else:
        return "Other"
    
def map_os_name(os):
    if os is None:
        return "Other"
    os = os.lower()
    if "android" in os:
        return "Android"
    elif "ios" in os:
        return "iOS"
    elif "freebsd" in os or "netbsd" in os or "openbsd" in os or "linux" in os:
        return "Linux"
    elif "windows 2000" in os or "windows 95/98/nt (unspecified)" in os or "windows nt" in os or "windows server 2003" in os\
        or "windows vista" in os or "windows xp" in os or "windows 7" == os:
        return "Windows older"
    elif "windows 10" == os or "windows 8" == os:
        return "Windows 10"
    elif "macos x" == os:
        return "MacOS X"
    else:
        return "Other"

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

class Useragents(object):
    def __init__(self, spark):
        self.sessionstart_table = spark.table(databases.table_sessionstart)
        selected_table          = self.retrieve_browser_OS()
        #timed_table            = self.convert_timestamp_to_date(selected_table)
        #timed_table             = self.add_day_time(selected_table)
        cleaned_table           = self.clean_browser_name(selected_table)
        self.cleaned_table      = self.map_browser_os(cleaned_table)
              
    def retrieve_browser_OS(self):
        return self.sessionstart_table.select(self.sessionstart_table.sessionnumber,
                                              self.sessionstart_table.devicebrowsername.alias("browser"),
                                              self.sessionstart_table.deviceplatformname.alias("os"),
                                              self.sessionstart_table.eventtimestamp.alias("sessionstarttime"))
    
    def clean_browser_name(self,df):
        return df.withColumn('browser', regexp_replace('browser', '\:(.*)', ''))
    
    def convert_timestamp_to_date(self,df):
        """
        TODO find out if necassary to convert to timestamp, i.e. what is the servers timestamp
        CET : central eastern time : UTC + 1
        CEST: central eastern summer time : UTC + 2 
        """
        utc_to_time_udf = udf(utc_to_time)
        df.show(10)
        UTC = df.withColumn("sessionstarttime", utc_to_time_udf("sessionstarttime"))
        #print("SHOULD BE IN LOCAL TIME")
        UTC.show(10)
        return UTC.withColumn("sessionstarttime", from_utc_timestamp("sessionstarttime", "CET")) #CEST?

    def add_day_time(self,df):
        get_weekday_udf = udf(get_weekday)
        get_hour_udf = udf(get_hour)
        get_minute_udf = udf(get_minute)
        df = df.withColumn("sessionstarttime_weekday", get_weekday_udf(df['sessionstarttime']))
        df = df.withColumn("sessionstarttime_hour", get_hour_udf(df['sessionstarttime']))
        df = df.withColumn("sessionstarttime_minute", get_minute_udf(df['sessionstarttime']))
        return df
    
    def map_browser_os(self, df):
        browser_mapper_udf = udf(map_browser_name)
        os_mapper_udf      = udf(map_os_name)
        df = df.withColumn("browser", browser_mapper_udf(df["browser"]))
        df = df.withColumn("os", os_mapper_udf(df["os"]))
        return df
        
    def run(self):
        return self.cleaned_table