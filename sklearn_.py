import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor, BayesianRidge, LinearRegression, Ridge, Lasso, Lars, LassoLars, TweedieRegressor, Perceptron, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, VotingRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # regression metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # classification metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold, validation_curve, learning_curve

from scipy.stats import pearsonr, spearmanr, randint
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,scale,normalize,minmax_scale, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


# load data
xl = pd.ExcelFile('summary_5min_period.xlsx')
lst_days_need_to_model = [xl.sheet_names[i] for i in [2,3,4,15,16,17,22,24,30]]

# create model & fitting
polynomial_features = PolynomialFeatures(degree=2)
plRegressor = LinearRegression(fit_intercept=True)
model = make_pipeline(polynomial_features, plRegressor)
lst_ml_class = [Ridge(),
				Lasso(),
				TweedieRegressor(),
				PassiveAggressiveRegressor(),
				RANSACRegressor(),
				TheilSenRegressor(),
				HuberRegressor(),
				QuantileRegressor(),
				SGDRegressor(),
				BayesianRidge(),
				LinearRegression(),
				model,
				DecisionTreeRegressor(),
				SVR(kernel='rbf'),
				SVR(kernel='linear'),
				SVR(kernel='poly', degree=2),
				RandomForestRegressor(),
				GradientBoostingRegressor(),
				AdaBoostRegressor(),
				VotingRegressor(estimators=[('sgd', SGDRegressor()),
										   ('bayes', BayesianRidge()),
										   ('linear', LinearRegression()),
										   ('polyLinear', make_pipeline(polynomial_features, plRegressor)),
										   ('deciTree', DecisionTreeRegressor()),
										   ('svr_rbf', SVR(kernel='rbf')),
										   ('svr_lin', SVR(kernel='linear')),
										   ('svr_poly', SVR(kernel='poly', degree=2)),
										   ('randForest', RandomForestRegressor()),
										   ('gradBoost', GradientBoostingRegressor()),
										   ('adaBoost', AdaBoostRegressor()),
										   ('ridge', Ridge()),
										   ('lasso', Lasso()),
										   ('tweed', TweedieRegressor()),
										   ('passive', PassiveAggressiveRegressor()),
										   ('ransac', RANSACRegressor()),
										   ('theilseng', TheilSenRegressor()),
										   ('huber', HuberRegressor()),
										   ('quantile', QuantileRegressor())]),
				KernelRidge(),
				KNeighborsRegressor(),
				BaggingRegressor(base_estimator=SGDRegressor()),
				BaggingRegressor(base_estimator=Ridge()),
				BaggingRegressor(base_estimator=Lasso()),
				BaggingRegressor(base_estimator=TweedieRegressor()),
				BaggingRegressor(base_estimator=PassiveAggressiveRegressor()),
				BaggingRegressor(base_estimator=TheilSenRegressor()),
				BaggingRegressor(base_estimator=HuberRegressor()),
				BaggingRegressor(base_estimator=QuantileRegressor()),
				BaggingRegressor(base_estimator=BayesianRidge()),
				BaggingRegressor(base_estimator=LinearRegression()),
				BaggingRegressor(base_estimator=model),
				BaggingRegressor(base_estimator=DecisionTreeRegressor()),
				BaggingRegressor(base_estimator=SVR(kernel='rbf')),
				BaggingRegressor(base_estimator=SVR(kernel='linear')),
				BaggingRegressor(base_estimator=SVR(kernel='poly', degree=2)),
				BaggingRegressor(base_estimator=RandomForestRegressor()),
				BaggingRegressor(base_estimator=GradientBoostingRegressor()),
				BaggingRegressor(base_estimator=AdaBoostRegressor())]
lst_ml_class_name = ['Ridge',
					'Lasso',
					'TweedieRegressor',
					'PassiveAggressiveRegressor',
					'RANSACRegressor',
					'TheilSenRegressor',
					'HuberRegressor',
					'QuantileRegressor',
					'SGDRegressor',
					'BayesianRidge',
					'LinearRegression',
					'Polynomial Regression',
					'DecisionTreeRegressor',
					'SVR(kernel=rbf)',
					'SVR(kernel=linear)',
					'SVR(kernel=poly, degree=2)',
					'RandomForestRegressor',
					'GradientBoostingRegressor',
					'AdaBoostRegressor',
					'VotingRegressor',
					'KernelRidge',
					'KNeighborsRegressor',
					'BaggingRegressor(base_estimator=SGDRegressor())',
					'BaggingRegressor(base_estimator=Ridge())',
					'BaggingRegressor(base_estimator=Lasso())',
					'BaggingRegressor(base_estimator=TweedieRegressor())',
					'BaggingRegressor(base_estimator=PassiveAggressiveRegressor())',
					'BaggingRegressor(base_estimator=RANSACRegressor())',
					'BaggingRegressor(base_estimator=TheilSenRegressor())',
					'BaggingRegressor(base_estimator=HuberRegressor())',
					'BaggingRegressor(base_estimator=QuantileRegressor())',
					'BaggingRegressor(base_estimator=BayesianRidge())',
					'BaggingRegressor(base_estimator=LinearRegression())',
					'BaggingRegressor(base_estimator=Polynomial Regression)',
					'BaggingRegressor(base_estimator=DecisionTreeRegressor())',
					'BaggingRegressor(base_estimator=SVR(kernel=rbf))',
					'BaggingRegressor(base_estimator=SVR(kernel=linear))',
					'BaggingRegressor(base_estimator=SVR(kernel=poly, degree=2))',
					'BaggingRegressor(base_estimator=RandomForestRegressor())',
					'BaggingRegressor(base_estimator=GradientBoostingRegressor())',
					'BaggingRegressor(base_estimator=AdaBoostRegressor())']

def fit_function(ml_class, ml_class_name, x_tr, y_tr, x_ts, y_ts):
	global dict_input_to_evaluate
	reg = ml_class
	reg.fit(x_tr,y_tr.values.ravel())
	y_train_predict = reg.predict(x_tr)
	y_test_predict = reg.predict(x_ts)
	dict_input_to_evaluate[ml_class_name] = {'Training':(y_tr,y_train_predict), 'Validation':(y_ts,y_test_predict)}


if __name__ == '__main__':
	df_results = pd.ExcelWriter('evaluate_results_daily.xlsx',engine='openpyxl')
	for each_day in lst_days_need_to_model:
		df = xl.parse(each_day)
		df['VC_ratio'] = df[['total_vol_all_lanes']].div(561.25, axis=0)
		x = df.loc[:, ['VC_ratio']]**2
		y = df.loc[:, ['travelTime']]
		# train & test set
		seed = 14
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=seed)
		dict_input_to_evaluate = {}
		for i, j in zip(lst_ml_class, lst_ml_class_name):
			fit_function(i, j, x_train, y_train, x_test, y_test)

		# evaluate model
		dict_evaluate_result = {}
		for each_model, train_test in dict_input_to_evaluate.items():
			dict_evaluate_result[each_model] = {}
			for each_type, tupl in train_test.items():
				mean_absolute_err = mean_absolute_error(*tupl)
				mean_squared_err = mean_squared_error(*tupl)
				r2_scor = r2_score(*tupl)
				if r2_scor < 0:
					print(each_day, each_model)
					break
				else:
					dict_evaluate_result[each_model][(each_type, 'R^2')] = r2_scor
					dict_evaluate_result[each_model][(each_type, 'MSE')] = mean_squared_err



		df_result = pd.DataFrame.from_dict(dict_evaluate_result, orient='index').round(3)
		df_result['Stability'] = df_result[df_result.columns[0]]/df_result[df_result.columns[2]]
		df_result.sort_values(by='Stability', ascending=True, inplace=True)
		df_result.to_excel(df_results, sheet_name=each_day)
	df_results.save()
