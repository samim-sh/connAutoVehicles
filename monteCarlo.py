import re
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor, BayesianRidge, LinearRegression, Ridge, Lasso, TweedieRegressor, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor, QuantileRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,BaggingRegressor,RandomForestRegressor,VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.mixture import GaussianMixture as GMM
import scipy.stats as stats
import os
import math
from collections import defaultdict


def random_sample_data(dist_type, params, typ, s):
    if 'Log-normal' in dist_type:
        mu, sig = map(float, re.findall(r'\d+\.\d+', params))
        sample = np.random.lognormal(mean=mu, sigma=sig, size=s)

    # elif 'Beta' in dist_type:
    #     a, b, loc, scale = stats.beta.fit(d)
    #     # alpha, beta, loc, scale = map(float, re.findall(r'\d+\.\d+', params))
    #     sample = stats.beta.rvs(a, b, loc, scale, size=s)

    elif 'Weibull' in dist_type:
        c, lam = map(float, re.findall(r'\d+\.\d+', params))
        sample = stats.weibull_min.rvs(c, loc=0, scale=lam, size=s)

    elif 'Gamma' in dist_type:
        shape, scale = map(float, re.findall(r'\d+\.\d+', params))
        sample = np.random.gamma(shape=shape, scale=scale, size=s)

    # elif 'Fisher-Tippett' in dist_type:
    #     beta, mu = map(float, re.findall(r'\d+\.\d+', params))
    #     sample = np.random.gumbel(loc=mu, scale=beta, size=s)

    elif 'Logistic' in dist_type:
        loc, scale = map(float, re.findall(r'\d+\.\d+', params))
        sample = np.random.logistic(loc=loc, scale=scale, size=s)

    elif 'GEV' in dist_type:
        c, scale, loc = map(float, re.findall(r'\d+\.\d+', params))
        sample = stats.genextreme.rvs(c, loc=loc, scale=scale, size=s)

    elif 'Normal' in dist_type:
        loc, scale = map(float, re.findall(r'\d+\.\d+', params))
        sample = np.random.normal(loc=loc, scale=scale, size=s)

    elif 'Exponential' in dist_type:
        scale = float(re.findall(r'\d+\.\d+', params)[0])
        sample = stats.expon.rvs(loc=0,scale=1/scale, size=s)

    elif 'Erlang' in dist_type:
        a, scale = map(float, re.findall(r'\d+\.\d+', params))
        sample = stats.erlang.rvs(a,loc=0, scale=1/scale, size=s)

    elif 'Poisson' in dist_type:
        lam = float(re.findall(r'\d+\.\d+', params)[0])
        sample = np.random.poisson(lam=lam, size=s)

    else:
        if typ == 'timeHeadway':
            gmm = GMM(n_components=3)
            gmm.fit(df_speed_hour.timeHeadway.values.reshape(-1, 1))
            sample = gmm.sample(n_samples=s)[0].reshape(-1)
        else:
            gmm = GMM(n_components=3)
            gmm.fit(df_speed_hour.Speed.values.reshape(-1, 1))
            sample = gmm.sample(s)[0].reshape(-1)
    # if P_c == .1:
    #     if typ == 'timeHeadway':
    #         sns.histplot(sample, kde=True, ax=ax_sample_time)
    #         ax_sample_time.set_title(f'{P_c} {day_date} {start_peak} {lane_id} {dist_type}')
    #         fig_time.tight_layout()
    #         fig_time.savefig(f'holy_moly/{day_date}/{lane_id}/timeHeadway {start_peak} {dist_type}.png')
    #     else:
    #         sns.histplot(sample, kde=True, ax=ax_sample_speed)
    #         ax_sample_speed.set_title(f'{P_c} {day_date} {start_peak} {lane_id} {dist_type}')
    #         fig_speed.tight_layout()
    #         fig_speed.savefig(f'holy_moly/{day_date}/{lane_id}/speed {start_peak} {dist_type}.png')
    #     plt.close()
    #
    # return sample


def plot_iteration(typ, lst):
    plt.xlabel('Iterations')
    plt.ylabel(typ, rotation=0)
    plt.title(f'{P_c} {day_date} {start_peak} {lane_id}')
    plt.plot(lst)
    plt.savefig(f'holy_moly/{typ}/{P_c} {day_date} {start_peak} {lane_id}.png')
    plt.close()


def ml_fit():
    # ML
    polynomial_features = PolynomialFeatures(degree=2)
    plRegressor = LinearRegression(fit_intercept=True)
    lst_reg = {k: v for k, v in zip({i.split('_')[0]:ind for ind,i in enumerate(xl_dist_headway.sheet_names)}.keys(),
                                    [BaggingRegressor(base_estimator=GradientBoostingRegressor()),
                                     KNeighborsRegressor(),
                                     BaggingRegressor(base_estimator=GradientBoostingRegressor()),
                                     BaggingRegressor(base_estimator=GradientBoostingRegressor()),
                                     VotingRegressor(estimators=[('sgd', SGDRegressor()),
                                                                 ('bayes', BayesianRidge()),
                                                                 ('linear', LinearRegression()),
                                                                 ('polyLinear',
                                                                  make_pipeline(polynomial_features,
                                                                                plRegressor)),
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
                                     BaggingRegressor(base_estimator=GradientBoostingRegressor()),
                                     AdaBoostRegressor(),
                                     BaggingRegressor(base_estimator=GradientBoostingRegressor()),
                                     BaggingRegressor(base_estimator=GradientBoostingRegressor())])
               }

    df_day_date['VC_ratio'] = df_day_date[['total_vol_all_lanes']].div(561.25, axis=0)
    x = df_day_date.loc[:, ['VC_ratio']] ** 2
    y = df_day_date.loc[:, ['travelTime']]
    # train & test set
    reg = lst_reg[day_date]
    reg.fit(x.values, y.values.ravel())
    return reg


if __name__ == '__main__':
    current_dir = os.getcwd()
    os.makedirs(f'{current_dir}/holy_moly/t', exist_ok=True)
    os.makedirs(f'{current_dir}/holy_moly/c', exist_ok=True)
    os.makedirs(f'{current_dir}/holy_moly/v_c', exist_ok=True)
    os.makedirs(f'{current_dir}/holy_moly/k', exist_ok=True)
    sample_size = 5000
    montecarlo_iteration = 1000

    spacing_headway = 2
    vehicle_length = 4.2
    xl = pd.ExcelFile('datasets/summary_5min_period.xlsx')
    xl_speed = pd.ExcelFile('datasets/speed_chosen.xlsx')

    xl_write = pd.ExcelWriter('holy_moly/final_result.xlsx', engine='openpyxl')
    xl_dist_headway = pd.ExcelFile('datasets/dist_headway.xlsx')
    """
    a = [pd.Series(xl_dist_headway.parse(each_sheet).Parameters.values,index=xl_dist_headway.parse(each_sheet).Distribution).to_dict() for each_sheet in xl_dist_headway.sheet_names]
    {i:j for b in a for i,j in b.items()}
    
    [('****', '****'), 
    ('Beta (4)', 'α= 2.193, β= 76.871, c= 2.019, d= 460.164'), 
    ('Erlang', 'K= 3.000, λ= 0.257'), 
    ('Exponential', 'λ=0.006'), 
    ('Fisher-Tippett (2)', 'β= 6.000, µ= 9.668'), 
    ('GEV', 'k= - 0.025, β= 26.297, µ= 49.536'), 
    ('Gamma (2)', 'k= 1.492, β= 5.652'), 
    ('Log-normal', 'µ= 1.856, σ= 0.910'), 
    ('Logistic', 'µ= 65.217, s= 19.606'), 
    ('Normal', 'µ= 200.000, σ= 147.132'), 
    ('Weibull (2)', 'β= 1.166, γ= 7.293')]"""

    xl_dist_speed = pd.ExcelFile('datasets/dist_speed.xlsx')
    """
    [('****', '****'),
    ('Beta (4)', 'α= 1.902, β= 5.120, c=80.505, d= 134.580'),
    ('Erlang', 'k=169.000, λ= 0.600'),
    ('Fisher-Tippett (2)', 'β= 5.060, µ= 72.820'),
    ('GEV', 'k= -0.054, β= 7.900, µ = 87.623'),
    ('Gamma (2)', 'k=197.32, β= 0.431'),
    ('Log-normal', 'µ= 4.550, σ= 0.084'),
    ('Logistic', 'µ= 90.743, s= 5.077'),
    ('Normal', 'µ= 93.174, σ= 9.014'),
    ('Poisson', 'λ= 77.269'),
    ('Weibull (2)', 'β= 4.329, γ= 84.120')]
    """

    for P_c in np.round(np.arange(0.1, 1, .1).tolist(), 2)[:]:
        dict_summary_table = {}
        df_writer = pd.DataFrame()
        lst_summary_table = defaultdict(lambda: defaultdict(dict))
        print('P_c', dt.datetime.now(), P_c)
        day_before = ''
        for each_sheet in xl_dist_headway.sheet_names[:]:

            print('each_sheet', dt.datetime.now(), each_sheet)
            day_date, lane_id = each_sheet.split('_')
            os.makedirs(f'{current_dir}/holy_moly/{day_date}/{lane_id}', exist_ok=True)
            dict_summary_table[day_date] = {i:dict() for i in range(1,6)}
            df_dist_table_headway = xl_dist_headway.parse(each_sheet, parse_dates=['Time', 'To'])
            df_dist_table_speed = xl_dist_speed.parse(each_sheet, parse_dates=['Time', 'To'])
            df_day_date = xl.parse(day_date.split('_')[0])
            if day_date == day_before:
                # print('pass', day_date, day_before)
                day_before = day_date
            else:
                # print(day_date, day_before)
                day_before = day_date
                # ML fitting
                reg = ml_fit()
            for indx_row, each_hour in df_dist_table_headway[:].iterrows():
                start_peak, end_peak = each_hour.Time.time(), each_hour.To.time()
                print('start_peak', start_peak)
                if end_peak == dt.time(0,0,0):
                    end_peak = dt.time(23,59,59)
                df_speed = xl_speed.parse(f'{day_date}_laneId_{lane_id}', usecols=['DateTime', 'timeHeadway', 'Speed'])
                df_speed_hour = df_speed[(df_speed.DateTime.dt.time > dt.datetime.combine(dt.date(1,1,1),start_peak).time()) &
                                         (df_speed.DateTime.dt.time <= end_peak)]
                N_mix = len(df_speed_hour)
                # if P_c == .1:
                #     fig_time, (ax_real_time, ax_sample_time) = plt.subplots(1,2,figsize=(16,9))
                #     fig_speed, (ax_real_speed, ax_sample_speed) = plt.subplots(1,2,figsize=(16,9))
                #     sns.histplot(df_speed_hour.timeHeadway, kde=True, ax=ax_real_time)
                #     sns.histplot(df_speed_hour.Speed, kde=True, ax=ax_real_speed)

                dist_type = df_dist_table_headway.at[indx_row, 'Distribution']
                params = df_dist_table_headway.at[indx_row,'Parameters']
                timeHeadway_sample_hh = random_sample_data(dist_type, params, 'timeHeadway', sample_size)
                timeHeadway_sample_cc = np.random.triangular(.3, .9, 1.5, size=sample_size)
                timeHeadway_sample_hc = np.random.triangular(.6, .9, 2.2, size=sample_size)
                timeHeadway_sample_ch = np.random.triangular(.45, .9, 2, size=sample_size)

                dist_type = df_dist_table_speed.at[indx_row, 'Distribution']
                params = df_dist_table_speed.at[indx_row, 'Parameters']
                speed_sample = random_sample_data(dist_type, params, 'speed', sample_size)
                N_c = round(P_c * N_mix)
                if N_c == 0:
                    N_c = math.ceil(P_c * N_mix)
                P_h = 1 - P_c
                N_h = round(P_h * N_mix)
                if N_h == 0:
                    N_h = math.ceil(P_h * N_mix)
                lst_N_mix = ['c']*N_c + ['h']*N_h
                n_cc, n_ch = 0, 0
                while (n_cc + n_ch) == 0:
                    np.random.shuffle(lst_N_mix)
                    df_car_queue = pd.DataFrame({'queue':lst_N_mix})
                    df_car_queue['queue_shift_neg_1'] = df_car_queue.shift(-1)
                    n_cc = sum((df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='c'))
                    n_hh = sum((df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='h'))
                    n_hc = sum((df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='c'))
                    n_ch = sum((df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='h'))
                    # if (n_cc + n_ch)==0:
                    #     print(f'N_mix {N_mix} - N_c {N_c} - N_h {N_h} - n_cc {n_cc} - n_ch {n_ch}')
                P_I = n_cc / (n_cc + n_ch)
                P_HH = n_hh / (n_hh + n_hc)
                # ρ_mn
                df_car_queue['P_mn'] = None
                df_car_queue.loc[(df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='h'),'P_mn'] = P_h * (1 - P_I)  #ch
                df_car_queue.loc[(df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='c'),'P_mn'] = 1 - P_h * (1 - P_I)  #cc
                df_car_queue.loc[(df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='c'),'P_mn'] = P_c * (1 - P_I)  #hc
                df_car_queue.loc[(df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='h'),'P_mn'] = 1 - P_c * (1 - P_I)  #hh
                # h_mn
                df_car_queue['h_mn'] = None
                # h_cc_sample, h_hh_sample, h_ch_sample, h_hc_sample = np.array([10]), np.array([5]), np.array([7]), np.array([9]),
                # while any(np.any(e_e >= h_hh_sample) for e_e in h_cc_sample) or\
                    # any(np.any(e_e <= h_cc_sample) for e_e in h_ch_sample) or\
                    # any(np.any(e_e <= h_hc_sample) for e_e in h_hh_sample):
                h_cc_sample = np.random.choice(timeHeadway_sample_cc, n_cc, replace=False)
                h_hc_sample = np.random.choice(timeHeadway_sample_hc, n_hc, replace=False)
                h_ch_sample = np.random.choice(timeHeadway_sample_ch, n_ch, replace=False)
                h_hh_sample = np.random.choice(timeHeadway_sample_hh, n_hh, replace=False)
                    # print('1')
                df_car_queue.loc[(df_car_queue.queue == 'c') & (df_car_queue.queue_shift_neg_1 == 'c'), 'h_mn'] = h_cc_sample  # cc
                df_car_queue.loc[(df_car_queue.queue == 'h') & (df_car_queue.queue_shift_neg_1 == 'c'), 'h_mn'] = h_hc_sample  # hc
                df_car_queue.loc[(df_car_queue.queue == 'c') & (df_car_queue.queue_shift_neg_1 == 'h'), 'h_mn'] = h_ch_sample  # ch
                df_car_queue.loc[(df_car_queue.queue == 'h') & (df_car_queue.queue_shift_neg_1 == 'h'), 'h_mn'] = h_hh_sample  # hh
                k_mixed_critical = 1 / (100 * P_c * (df_car_queue.P_mn*df_car_queue.h_mn).sum() + spacing_headway * (1 - P_c) + vehicle_length)
                k_mixed_jam = 1 / (spacing_headway * (1 - P_c) + vehicle_length)

                # monteCarlo for K-mixed
                dict_k = {}
                lst_k_results = []
                sum_k = 0
                for iterate in range(montecarlo_iteration):
                    df_car_queue['h_mn_kMonteCarlo'] = None
                    # h_cc_sample, h_hh_sample, h_ch_sample, h_hc_sample = np.array([10]),np.array([5]),np.array([7]),np.array([9]),
                    # while any(np.any(e_e >= h_hh_sample) for e_e in h_cc_sample) or \
                    #         any(np.any(e_e <= h_cc_sample) for e_e in h_ch_sample) or \
                    #         any(np.any(e_e <= h_hc_sample) for e_e in h_hh_sample):
                    h_cc_sample = np.random.choice(timeHeadway_sample_cc, n_cc, replace=False)
                    h_hc_sample = np.random.choice(timeHeadway_sample_hc, n_hc, replace=False)
                    h_ch_sample = np.random.choice(timeHeadway_sample_ch, n_ch, replace=False)
                    h_hh_sample = np.random.choice(timeHeadway_sample_hh, n_hh, replace=False)
                        # print('2')
                    df_car_queue.loc[(df_car_queue.queue == 'c') & (df_car_queue.queue_shift_neg_1 == 'c'), 'h_mn_kMonteCarlo'] = h_cc_sample  # cc
                    df_car_queue.loc[(df_car_queue.queue == 'h') & (df_car_queue.queue_shift_neg_1 == 'c'), 'h_mn_kMonteCarlo'] = h_hc_sample  # hc
                    df_car_queue.loc[(df_car_queue.queue == 'c') & (df_car_queue.queue_shift_neg_1 == 'h'), 'h_mn_kMonteCarlo'] = h_ch_sample  # ch
                    df_car_queue.loc[(df_car_queue.queue == 'h') & (df_car_queue.queue_shift_neg_1 == 'h'), 'h_mn_kMonteCarlo'] = h_hh_sample  # hh
                    k = 1 / (np.random.choice(speed_sample, replace=False) * (P_c * (df_car_queue.P_mn*df_car_queue.h_mn_kMonteCarlo).sum()) + spacing_headway * P_h + vehicle_length)
                    sum_k += k
                    avg_k = sum_k / (iterate + 1)
                    lst_k_results.append(avg_k)
                plot_iteration('k', lst_k_results)
                if k_mixed_jam > lst_k_results[-1] > k_mixed_critical:
                    # monteCarlo for C-mixed
                    dict_t, dict_c, dict_v_c = {}, {}, {}
                    lst_t_results, lst_c_results, lst_v_c_results = [], [], []
                    sum_t_predicted, sum_c_predicted, sum_v_c_predicted,  = 0, 0, 0
                    for iterate in range(montecarlo_iteration):
                        h_hh = np.random.choice(timeHeadway_sample_hh, replace=False)
                        h_hc = np.random.choice(timeHeadway_sample_hc, replace=False)
                        h_ch = np.random.choice(timeHeadway_sample_ch, replace=False)
                        h_cc = np.random.choice(timeHeadway_sample_cc, replace=False)
                        if (h_cc, h_ch, h_hc, h_hh) not in dict_t:
                            C_m = 3600 / (P_c * P_I * h_cc + P_c * (1 - P_I) * h_ch + P_h * (1 - P_HH) * h_hc + P_h * P_HH * h_hh)
                            v_c_ratio = (N_mix / C_m)**2
                            t_predicted = reg.predict([[v_c_ratio]])[0]
                            dict_c[(h_cc, h_ch, h_hc, h_hh)] = C_m
                            dict_v_c[(h_cc, h_ch, h_hc, h_hh)] = v_c_ratio
                            dict_t[(h_cc, h_ch, h_hc, h_hh)] = t_predicted
                        else:
                            t_predicted = dict_t[(h_cc, h_ch, h_hc, h_hh)]
                            C_m = dict_c[(h_cc, h_ch, h_hc, h_hh)]
                            v_c_ratio = dict_v_c[(h_cc, h_ch, h_hc, h_hh)]
                        sum_t_predicted += t_predicted
                        sum_c_predicted += C_m
                        sum_v_c_predicted += v_c_ratio
                        avg_t_predicted = sum_t_predicted / (iterate + 1)
                        avg_c_predicted = sum_c_predicted / (iterate + 1)
                        avg_v_c_predicted = sum_v_c_predicted / (iterate + 1)
                        lst_t_results.append(avg_t_predicted)
                        lst_c_results.append(avg_c_predicted)
                        lst_v_c_results.append(avg_v_c_predicted)
                    plot_iteration('t', lst_t_results)
                    plot_iteration('c', lst_c_results)
                    plot_iteration('v_c', lst_v_c_results)
                    lst_summary_table[day_date][lane_id][start_peak] = [lst_t_results[-1], lst_c_results[-1], lst_v_c_results[-1]]
                else:
                    # print(f'{P_c} {day_date} {start_peak} {lane_id}', k_mixed_critical, lst_k_results[-1], k_mixed_jam)
                    lst_summary_table[day_date][lane_id][start_peak] = [None,None,None]
        for day in lst_summary_table:
            df_tmp = pd.DataFrame()
            for lan in lst_summary_table[day]:
                tups = [i for b in [[(day, x, 't'), (day, x, 'c'), (day, x, 'v_c')] for x in lst_summary_table[day][lan].keys()] for i in b]
                colums = pd.MultiIndex.from_tuples(tups, names=['date level 1', 'hour level 2', 'var level 3'])
                df_summary_table = pd.DataFrame([[i for b in lst_summary_table[day][lan].values() for i in b]], columns=colums, index=[lan])
                df_tmp = pd.concat([df_tmp, df_summary_table], axis=0)
            df_writer = pd.concat([df_writer, df_tmp], axis=1)
        df_writer.to_excel(xl_write, sheet_name=f'{P_c}')
    xl_write.save()