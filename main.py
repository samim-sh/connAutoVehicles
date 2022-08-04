import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import jdatetime as jdt
import math
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler


def standarise(column,pct,pct_lower):
	sc = StandardScaler()
	y = df_speed[column][df_speed[column].notnull()].to_list()
	y.sort()
	len_y = len(y)
	y = y[int(pct_lower * len_y):int(len_y * pct)]
	len_y = len(y)
	yy=([[x] for x in y])
	sc.fit(yy)
	y_std =sc.transform(yy)
	y_std = y_std.flatten()
	return y_std,len_y,y


def fit_distribution(column, pct, pct_lower):
	# Set up list of candidate distributions to use
	# See https://docs.scipy.org/doc/scipy/reference/stats.html for more
	y_std, size, y_org = standarise(column, pct, pct_lower)
	dist_names = ['weibull_min', 'norm', 'weibull_max', 'beta',
				  'invgauss', 'uniform', 'gamma', 'expon', 'lognorm', 'pearson3', 'triang']

	chi_square_statistics = []
	# 11 bins
	percentile_bins = np.linspace(0, 100, 11)
	percentile_cutoffs = np.percentile(y_std, percentile_bins)
	observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
	cum_observed_frequency = np.cumsum(observed_frequency)

	# Loop through candidate distributions

	for distribution in dist_names:
		# Set up distribution and get fitted distribution parameters
		dist = getattr(stats, distribution)
		param = dist.fit(y_std)
		print("{}\n{}\n".format(dist, param))

		# Get expected counts in percentile bins
		# cdf of fitted sistrinution across bins
		cdf_fitted = dist.cdf(percentile_cutoffs, *param)
		expected_frequency = []
		for bin in range(len(percentile_bins) - 1):
			expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
			expected_frequency.append(expected_cdf_area)

		# Chi-square Statistics
		expected_frequency = np.array(expected_frequency) * size
		cum_expected_frequency = np.cumsum(expected_frequency)
		ss = round(sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency), 0)
		chi_square_statistics.append(ss)

	# Sort by minimum ch-square statistics
	results = pd.DataFrame()
	results['Distribution'] = dist_names
	results['chi_square'] = chi_square_statistics
	results.sort_values(['chi_square'], inplace=True)

	print('\nDistributions listed by Betterment of fit:')
	print('............................................')
	print(results)

writer_speed = pd.ExcelWriter('speed_.xlsx', engine='openpyxl')
writer_volume = pd.ExcelWriter('volume_.xlsx', engine='openpyxl')
writer_travelTime = pd.ExcelWriter('travelTime_.xlsx', engine='openpyxl')

for laneId in [1,2,3,4,5]:
	df_speed = pd.read_csv('datasets/code-۱۱۳۴۲۸.csv')
	df_speed = df_speed[(df_speed.LaneID==laneId)&(df_speed.Class==1)].reset_index(drop=True)
	df_speed['Date Time'] = pd.to_datetime(df_speed['Date Time'], yearfirst=True, errors='coerce')
	df_speed.rename(columns = {'Date Time':'DateTime'}, inplace=True)
	df_speed['date_'] = df_speed.DateTime.dt.date
	df_speed = df_speed[(df_speed.DateTime>=dt.datetime(2020,8,22,14,0,0))&(df_speed.DateTime<=dt.datetime(2020,9,22,0,0,0))].reset_index(drop=True)  #1399-6-15 : 1399-6-21
	year,month,day = df_speed.DateTime[0].year, df_speed.DateTime[0].month, df_speed.DateTime[0].day
	df_speed['timeHeadway'] = (df_speed.DateTime - df_speed.DateTime.shift(periods=1, fill_value=dt.datetime(year,month,day,0,10,0))).dt.total_seconds().round(2)
	df_speed.index = df_speed.DateTime
	df_volume = df_speed.resample("5T",label='right').agg(volume=('Speed','count'),avg_speed=('Speed','mean'),avg_timeHeadway=('timeHeadway','mean'),date_=('date_','first')).round(2)

	for date in df_speed.groupby('date_').size().index.values:
		df_speed[df_speed.date_==date].to_excel(writer_speed, sheet_name=f'{date}_laneId_{laneId}')
		df_volume[df_volume.date_==date].to_excel(writer_volume, sheet_name=f'{date}_laneId_{laneId}')
	writer_speed.save()
	writer_volume.save()
	# df_speed.to_excel(f'dfSpeed_{laneId}.xlsx',index=False)
	# df_volume.to_excel(f'dfVolume_{laneId}.xlsx',index=True)
	# fit_distribution('Speed', .99, .01)
	# mu = df_speed.Speed.mean()
	# sigma = df_speed.Speed.std()
	# plt.plot(df_speed.Speed.sort_values(), stats.norm.pdf(df_speed.Speed.sort_values(), mu, sigma))
	# plt.show()
	# df_speed['z_score'] = stats.zscore(df_speed.Speed)
	# df_speed = df_speed[(df_speed.z_score<2.576)&(df_speed.z_score>-2.576)]
	# fit_distribution('Speed', .99, .01)

df_time = pd.read_excel('datasets/99_06_MonthlyReport_1399_6_آزادراه_تهران_به_کرج_.xlsx', names=['roadName', 'travelTime', 'time_', 'date_'])
df_time['travelTime'] = df_time.travelTime.apply(lambda x: x.minute if x is not np.NaN else None)
df_time['date_'] = df_time.date_.apply(lambda x: jdt.datetime.togregorian(jdt.datetime.strptime(x,'%Y/%m/%d')).date())
df_time['DateTime'] = df_time.apply(lambda row: dt.datetime.combine(row.date_, row.time_),axis=1)
df_time = df_time[(df_time.DateTime>=dt.datetime(2020,8,22,14,0,0))&(df_time.DateTime<=dt.datetime(2020,9,22,0,0,0))].reset_index(drop=True)  #1399-6-1 : 1399-6-31
for column in ['travelTime']:
	for ind,row in df_time[[column]].iterrows():
		if ~np.isnan(row[column]):
			previous = row[column]
		else:
			indx = ind + 1
			while np.isnan(df_time.loc[indx,column]):
				indx+=1
			next = df_time.loc[indx,column]
			df_time[column][ind] = (previous+next)/2
			previous = df_time[column][ind]
for date in df_time.groupby('date_').size().index.values:
	df_time[df_time.date_ == date].to_excel(writer_travelTime, sheet_name=f'{date}')
writer_travelTime.save()
# df_time.to_excel('dfTime.xlsx',index=False)

# # IQR method
# q3, q1 = np.percentile(df_speed.Speed, [75 ,25])
# iqr = q3 - q1
# len(np.where(df_speed.Speed>q3+1.5*iqr)[0]), len(np.where(df_speed.Speed<q1-1.5*iqr)[0])
# # Z-score method
# df_speed['z_score'] = stats.zscore(df_speed.Speed)
# len(np.where(df_speed['z_score']>2.576)[0]),len(np.where(df_speed['z_score']<-2.576)[0])  # 99%
# df_speed = df_speed[(df_speed.z_score<2.576)&(df_speed.z_score>-2.576)]
