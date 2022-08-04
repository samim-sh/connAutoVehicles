import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# for i in [1,2,3,4,5]:
# 	df = pd.read_excel(f'dfSpeed_{i}.xlsx')
# 	df[df.timeHeadway<800][['timeHeadway']].plot.hist(bins='fd')
# 	plt.savefig(f'timeHeadway_{i}.png')
# 	df[['Speed']].plot.hist(bins='fd')
# 	plt.savefig(f'speed_{i}.png')
# df = pd.read_excel('dfTime.xlsx')
# df[['travelTime']].plot.hist(bins='fd')
# plt.savefig(f'travelTime.png')



xl_speed = pd.ExcelFile(f'speed.xlsx')
xl_travelTime = pd.ExcelFile(f'travelTime.xlsx')
xl_volume = pd.ExcelFile(f'volume.xlsx')
for i in xl_speed.sheet_names:
	df = xl_speed.parse(i)
	df[df.timeHeadway<800][['timeHeadway']].plot.hist(bins='fd')
	plt.savefig(f'timeHeadway/timeHeadway_{i}.png')
	df[['Speed']].plot.hist(bins='fd')
	plt.savefig(f'speed/speed_{i}.png')
for i in xl_travelTime.sheet_names:
	df = xl_travelTime.parse(i)
	df[['travelTime']].plot.hist(bins='fd')
	plt.savefig(f'travelTime/travelTime_{i}.png')
for i in xl_volume.sheet_names:
	df = xl_volume.parse(i)
	df[['volume']].plot.hist(bins='fd')
	plt.savefig(f'volume/volume_{i}.png')