import pandas as pd
import numpy as np


xl_volume = pd.ExcelFile('volume_.xlsx')
xl_travelTime = pd.ExcelFile('travelTime_.xlsx')
writer_travelTime_speed_5min_period = pd.ExcelWriter('summary_5min_period.xlsx', engine='openpyxl')


df_vol = {sheet_name.split('_')[0]: {each.split('_')[2]:each for each in [f"{sheet_name.split('_')[0]}_laneId_{n}" for n in range(1,6)]}
		  for sheet_name in xl_volume.sheet_names}
for each_day in df_vol:
	a = pd.concat([xl_volume.parse(each_day_lane,skiprows=1,header=None,names=['DateTime', f'volume_{each_lane}', f'avg_speed_{each_lane}', 'avg_timeHeadway', 'date_']).set_index('DateTime') for each_lane, each_day_lane in df_vol[each_day].items()], join='inner', axis=1)
	vol = a[[f'volume_{i}' for i in range(1,6)]]
	vol[[f'VC_ratio_{i}' for i in range(1, 6)]] = vol[[f'volume_{n}' for n in range(1, 6)]].div(112.25, axis=0)
	avg_speed = a[[f'avg_speed_{i}' for i in range(1,6)]]
	avg_speed['sigma_invert_speed'] = avg_speed.apply(lambda x:1/x).sum(axis=1)
	avg_time = xl_travelTime.parse(each_day,usecols=(lambda x: 'Unnamed' not in x))
	avg_time['DateTime'] = avg_time['DateTime'].dt.floor('T')
	time_and_invertSpeed = pd.concat([avg_time.set_index('DateTime'), avg_speed, vol], axis=1, join='inner')
	time_and_invertSpeed['x_length_km'] = time_and_invertSpeed.apply(lambda x:x.travelTime*5/(60*x.sigma_invert_speed), axis=1)
	time_and_invertSpeed[[f't_{i}' for i in range(1,6)]] = time_and_invertSpeed[[f'avg_speed_{i}' for i in range(1,6)]].rdiv(60*time_and_invertSpeed.x_length_km, axis=0)
	time_and_invertSpeed.to_excel(writer_travelTime_speed_5min_period, sheet_name=f'{each_day}')
writer_travelTime_speed_5min_period.save()
# 60*(.09475/(1/time_and_invertSpeed[[f'avg_speed_{i}' for i in range(1,6)]].iloc[0,:]).sum())/5