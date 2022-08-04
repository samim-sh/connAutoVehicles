import pprint
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import seaborn as sns
import datetime as dt


# xl = pd.ExcelFile('summary_5min_period.xlsx',)
# for day in xl.sheet_names:
# 	df = xl.parse(day)
# 	for lane in range(1,6):
# 		sns.lmplot(x=f'VC_ratio_{lane}', y=f't_{lane}', data=df, fit_reg=True)
# 		os.makedirs(f'/home/samim/pycharm_projects/paper/t_VCratio_scatter_plot/lane_{lane}',exist_ok=True)
# 		plt.savefig(f't_VCratio_scatter_plot/lane_{lane}/{day}.png')
# 		plt.close()


def dict_r2_by_date(direc,beta):
	dict_r2 = {i:{} for i in range(1,6)}
	for day in xl.sheet_names:
		df = xl.parse(day)
		for lane in range(1,6):
			X = df.loc[:, f'VC_ratio_{lane}'].values.reshape(-1, 1)**beta
			Y = df.loc[:, f't_{lane}'].values.reshape(-1, 1)
			linear_regressor = LinearRegression()
			linear_regressor.fit(X, Y)
			Y_pred = linear_regressor.predict(X)

			plt.scatter(X, Y)
			plt.plot(X, Y_pred, color='red')
			Coefficient_of_Determination = linear_regressor.score(X, Y)
			# coef = linear_regressor.coef_
			# intercept = linear_regressor.intercept_
			# plt.annotate(f"Coefficient_of_Determination (r^2): {Coefficient_of_Determination :.2f}", xy=(min(X), max(Y)))
			# plt.annotate(f'formula: {coef.item():.2f}x + {intercept.item():.2f}', xy=(min(X), max(Y)-.1*(max(Y)-min(Y))))
			#
			# os.makedirs(f'/home/samim/pycharm_projects/paper/{direc}/lane_{lane}', exist_ok=True)
			# plt.savefig(f'{direc}/lane_{lane}/{day}.png')
			# plt.close()
			dict_r2[lane][day] = Coefficient_of_Determination
	return dict_r2

def dict_r2_by_weekday(direc):
	dict_r2 = {i:{} for i in range(1,6)}
	days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
	dict_day = {k:None for k in days}

	for each_day in days:
		df = pd.concat([xl.parse(day,
								usecols=['DateTime', 'day_of_week'] + [f't_{n}' for n in range(1, 6)] + [f'VC_ratio_{n}' for
																										 n in range(1, 6)],
								ignore_index=True) for day in xl.sheet_names if days[dt.datetime.strptime(day, '%Y-%m-%d').weekday()]==each_day])
		dict_day[each_day] = df
		for lane in range(1,6):
			X = df.loc[:, f'VC_ratio_{lane}'].values.reshape(-1, 1)
			Y = df.loc[:, f't_{lane}'].values.reshape(-1, 1)
			linear_regressor = LinearRegression()
			linear_regressor.fit(X, Y)
			Y_pred = linear_regressor.predict(X)

			plt.scatter(X, Y)
			plt.plot(X, Y_pred, color='red')
			Coefficient_of_Determination = linear_regressor.score(X, Y)
			coef = linear_regressor.coef_
			intercept = linear_regressor.intercept_
			plt.annotate(f"Coefficient_of_Determination (r^2): {Coefficient_of_Determination :.2f}", xy=(min(X), max(Y)))
			plt.annotate(f'formula: {coef.item():.2f}x + {intercept.item():.2f}', xy=(min(X), max(Y)-.1*(max(Y)-min(Y))))

			os.makedirs(f'/home/samim/pycharm_projects/paper/{direc}/lane_{lane}', exist_ok=True)
			plt.savefig(f'{direc}/lane_{lane}/{days.index(each_day)+2 if days.index(each_day)<5 else days.index(each_day)-5}-{each_day}.png')
			plt.close()
			dict_r2[lane][each_day] = Coefficient_of_Determination
	return dict_r2



def plot_r(dict_value,direc,name):
	labels = dict_value[1].keys()
	lane_1 = dict_value[1].values()
	lane_2 = dict_value[2].values()
	lane_3 = dict_value[3].values()
	lane_4 = dict_value[4].values()
	lane_5 = dict_value[5].values()

	x = np.arange(0,len(labels)*2,2)  # the label locations
	width = .25  # the width of the bars
	fig, ax = plt.subplots(figsize=(16,9))
	rects1 = ax.bar(x - 2*width, lane_1, width=width, label='lane_1', align='edge')
	rects2 = ax.bar(x - width, lane_2, width=width, label='lane_2', align='edge')
	rects3 = ax.bar(x, lane_3, width=width, label='lane_3', align='edge')
	rects4 = ax.bar(x + width, lane_4, width=width, label='lane_4', align='edge')
	rects5 = ax.bar(x + 2*width, lane_5, width=width, label='lane_5', align='edge')
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel(name)
	ax.set_title(f'{name} by lanes')
	ax.set_xticks(x, labels, rotation='vertical')
	ax.legend()
	ax.bar_label(rects1, padding=4, fontsize=10, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects2, padding=4, fontsize=10, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects3, padding=4, fontsize=10, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects4, padding=4, fontsize=10, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects5, padding=4, fontsize=10, rotation='vertical', fmt='%.2f')
	fig.tight_layout()
	fig.savefig(f'{direc}/{name}.png')
	plt.close()

xl_beta_changing = pd.ExcelWriter('segment_length_changing/beta_r_squared.xlsx', engine='openpyxl')
xl_beta_25km = pd.ExcelWriter('segment_length_25km/beta_r_squared.xlsx', engine='openpyxl')
for beta in np.arange(1,7,.5):
	# _changing
	xl = pd.ExcelFile('summary_5min_period_changing.xlsx')
	direct = f'segment_length_changing/beta{beta}'
	dict_r2 = dict_r2_by_date(f'{direct}', beta)
	plot_r(dict_r2,'segment_length_changing',f'{beta}_Coefficient_of_Determination')
	# dict_r2 = dict_r2_by_weekday(f'{direct}/by_weekday')
	# plot_r(dict_r2,direct,'by_weekday_Coefficient_of_Determination')
	pd.DataFrame(dict_r2).round(2).to_excel(xl_beta_changing, sheet_name=f'{beta}')

	# _25km
	xl = pd.ExcelFile('summary_5min_period_25km.xlsx')
	direct = f'segment_length_25km/beta{beta}'
	dict_r2 = dict_r2_by_date(f'{direct}', beta)
	plot_r(dict_r2,'segment_length_25km',f'{beta}_Coefficient_of_Determination')
	# dict_r2 = dict_r2_by_weekday(f'{direct}/by_weekday')
	# plot_r(dict_r2,direct,'by_weekday_Coefficient_of_Determination')
	pd.DataFrame(dict_r2).round(2).to_excel(xl_beta_25km, sheet_name=f'{beta}')

xl_beta_changing.save()
xl_beta_25km.save()



# max_r2 = {a:max(b, key=b.get) for a, b in dict_r2.items()}
# for k,v in max_r2.items():
# 	print(k, v, dict_r2[k][v])
# def sorted_simple_dict(d):
# 	return {k: d[k] for k in sorted(d, key=d.get, reverse=True)}
# def sorted_once_nested_dict(d):
# 	return {k: sorted_simple_dict(v) for k, v in d.items()}
# sorted_dict_r2 = sorted_once_nested_dict(dict_r2)
# pprint.pprint(sorted_dict_r2, sort_dicts=False)






# m, b, r_value, p_value, std_err = stats.linregress(df[f'VC_ratio_{lane}'], df[f't_{lane}'])
#
# fig, ax = plt.subplots()
# ax.scatter(df[f'VC_ratio_{lane}'], df[f't_{lane}'])
# ax.plot(df[f'VC_ratio_{lane}'], m * df[f't_{lane}'] + b)
# ax.annotate(f"r^2: {r_value ** 2:.2f}", xy=(1950, 19500))
# ax.annotate(f'formula: {m:.2f}x + {b:.2f}', xy=(1950, 18500))