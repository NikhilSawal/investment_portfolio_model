import os
import datetime
import numpy as np
import pandas as pd
import psycopg2
import pandas.io.sql as psql
from typing import *

def get_data(start_date: str, end_date: str, company_name: str) -> pd.DataFrame:

    pg_auth = os.environ.get("PG_AUTH")

    conn = psycopg2.connect(host="localhost",
                            user="postgres",
                            dbname="investment_db",
                            password=pg_auth)


    df = psql.read_sql('''
                        select * from
                        	(
                        		select
                        			date_trunc('hour', date) date,
                        			price,
                        			delta_price,
                        			delta_price_perc,
                        			top_3_news,
                        			news_source
                        		from stock_price sp
                        		where
                        			name = '{}' and
                        			sp.date::date between timestamp '{}' and timestamp '{}'
                        	) a
                            left join
                            		(
                            			select
                            					date_trunc('hour', date) mi_date,
                            					snp_500,
                            					snp_500_delta,
                            					snp_500_delta_perc,
                            					dow_30,
                            					dow_30_delta,
                            					dow_30_delta_perc,
                            					nasdaq,
                            					nasdaq_delta,
                            					nasdaq_delta_perc
                            			from market_index
                            			where date::date between timestamp '{}' and timestamp '{}'
                            			order by 1
                            		) mi
                            on mi.mi_date = a.date
                            order by 1
                       '''.format(company_name, start_date, end_date, start_date, end_date), conn)

    # Remove duplicate date
    df = df.drop('mi_date', axis=1)

    # Create df of dates to account for missing data
    date_df = pd.DataFrame([i.strftime('%Y-%m-%d %H:%M:%S') for i in pd.date_range(start="'{}' 08:00:00".format(start_date), end="'{}' 16:00:00".format(end_date), freq='H') if (0 <= i.weekday() <= 4) & (8 <= i.hour <= 16)])
    date_df.columns = ['date']
    date_df['date'] = pd.to_datetime(date_df['date'])

    final_df = pd.merge(date_df, df, how='left', on=['date', 'date'])

    return final_df
