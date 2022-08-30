import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

xl = pd.ExcelFile('summary_5min_period.xlsx')
dict_df = {day: {k: v for k, v in
				 enumerate(xl.parse(day, usecols=['volume_1', 'volume_2', 'volume_3', 'volume_4', 'volume_5']).sum(),
						   start=1)} for day in xl.sheet_names}
df = pd.DataFrame.from_dict(dict_df)
labels = df.columns.to_list()
width = .33  # the width of the bars


def bar_chart_by_lanes():

	lane_1 = df.loc[1,:].to_list()
	lane_2 = df.loc[2,:].to_list()
	lane_3 = df.loc[3,:].to_list()
	lane_4 = df.loc[4,:].to_list()
	lane_5 = df.loc[5,:].to_list()

	x = np.arange(0,len(dict_df.keys())*2,2)  # the label locations
	fig, ax = plt.subplots(figsize=(16,9))
	rects1 = ax.bar(x - 2*width, lane_1, width=width, label='lane_1', align='edge')
	rects2 = ax.bar(x - width, lane_2, width=width, label='lane_2', align='edge')
	rects3 = ax.bar(x, lane_3, width=width, label='lane_3', align='edge')
	rects4 = ax.bar(x + width, lane_4, width=width, label='lane_4', align='edge')
	rects5 = ax.bar(x + 2*width, lane_5, width=width, label='lane_5', align='edge')


	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('volume')
	ax.set_xticks(x, labels, rotation='vertical')
	ax.legend()
	ax.bar_label(rects1, padding=4, fontsize=8, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects2, padding=4, fontsize=8, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects3, padding=4, fontsize=8, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects4, padding=4, fontsize=8, rotation='vertical', fmt='%.2f')
	ax.bar_label(rects5, padding=4, fontsize=8, rotation='vertical', fmt='%.2f')
	fig.tight_layout()
	fig.savefig('lane1_5.png')
	plt.show()
	plt.close()


def bar_chart_volume():
	lanes = df.sum()
	x = np.arange(0,len(dict_df.keys()))  # the label locations
	fig, ax = plt.subplots(figsize=(16,9))
	rects1 = ax.bar(x, lanes, width=width, label='lanes', align='edge')
	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('volume')
	ax.set_xticks(x, labels, rotation='vertical')
	ax.legend()
	ax.bar_label(rects1, padding=4, fontsize=8, rotation='vertical', fmt='%.2f')
	fig.tight_layout()
	fig.savefig('lanes.png')
	plt.show()
	plt.close()

def heatmap():
	fig, ax = plt.subplots(figsize=(16, 9))
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams["font.size"] = 11
	df_stacked = df.stack().reset_index()
	df_stacked.replace({k:indx for indx,k in enumerate(df.columns)})
	df_stacked.columns = ['lanes', 'date', 'Value']
	table = df_stacked.pivot('lanes', 'date', 'Value')
	ax = sb.heatmap(table, annot=True, fmt="d", linewidths=.2, cmap="YlGnBu", square=True, cbar_kws={'shrink':.3})
	ax.invert_yaxis()
	ax.set_xticks([i+.5 for i in np.arange(0,31,1)], labels, rotation='vertical')
	fig.savefig('volume_heatmap.png')
	plt.show()


def bar_chart_volume_bay_lanes_stacked():
	def valuelabel(labels, ax):
		for i in range(len(labels[1:])):
			if i in [6,7]:
				ax.text(i, ax.containers[-1][i].xy[1] + ax.containers[-1][i]._height + 1500, 'holiday', ha='center')
			elif i in [4, 5, 11, 12, 18, 19, 25, 26]:
				ax.text(i, ax.containers[-1][i].xy[1] + ax.containers[-1][i]._height + 1500, 'weekend', ha='center')
	# ------- Plot -------
	plt.rcParams["font.family"] = "Times New Roman"
	plt.rcParams["font.size"] = 11
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
	df.drop(columns='2020-08-22', inplace=True)
	df.index = [f'lane {i}' for i in range(1,6)]
	df.T.plot(ax=ax, kind='bar', stacked=True, color=['springgreen', 'greenyellow', 'yellow', 'orange', 'red'])
	ax.margins(.08)
	ax.bar_label(ax.containers[-1], fmt='%d')
	valuelabel(labels, ax)
	# x-axis
	ax.set_xticklabels(labels[1:])
	ax.set_xlabel('Date')
	ax.tick_params(axis='x', rotation=90)
	# y-axis
	ax.set_ylabel('Volume')
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1, prop={'size': 15})
	fig.savefig('volume_bar_chart.png')
	plt.show()
bar_chart_volume_bay_lanes_stacked()