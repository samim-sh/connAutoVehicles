import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import jdatetime as jdt


df_volume = pd.read_excel('datasets/volume/Hourly 113257 ‫آزادراه پرديس - تهران (بومهن)‬ .xlsx', skiprows=1, parse_dates=True, sheet_name=0 , usecols=range(2,16),
				   names=['start', 'end', 'operation_time', 'total_vehicle', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'avg_speed', 'speeding_count', 'unauthorized_spacing', 'illegal_overtaking', 'passenger_car_equivalent'])
df_volume['date_'] = df_volume.start.apply(lambda x: jdt.datetime.togregorian(jdt.datetime.strptime(x,'%Y/%m/%d %H:%M:%S')).date())
df_volume['time_'] = df_volume.start.apply(lambda x: jdt.datetime.togregorian(jdt.datetime.strptime(x,'%Y/%m/%d %H:%M:%S')).time())
df_volume.drop(['start', 'end'], axis=1, inplace=True)

df_time = pd.read_excel('datasets/time/Bomehen-Tehran99-7.xlsx', parse_dates=True, sheet_name=0 , usecols=[1,2,3], names=['travelTime', 'time_of_day', 'date_'])
df_time['travelTime'] = df_time.travelTime.apply(lambda x: x.minute if x is not np.NaN else None)
df_time['date_'] = df_time.date_.apply(lambda x: jdt.datetime.togregorian(jdt.datetime.strptime(x,'%Y/%m/%d')).date())
df_time['travelTime'] = (df_time.travelTime.ffill()+df_time.travelTime.bfill())/2
df_time['travelTime'] = df_time.travelTime.bfill().ffill()

print(df_time)
