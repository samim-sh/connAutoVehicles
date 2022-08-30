import re
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.mixture import GaussianMixture as GMM


def plot_peak_hour(each_day, db):

    db.loc[:,'time_'] = db['time_'].dt.floor('T')
    rolling_window = db.iloc[::-1].rolling('60T', closed='right', on='time_').mean().round(2)[::-1]
    value = rolling_window['travelTime'].max()
    start_peak_hour = rolling_window.loc[rolling_window['travelTime'].idxmax(), 'time_']
    end_peak_hour = start_peak_hour + np.timedelta64(5 * 12, 'm')
    peak_period = start_peak_hour.time(), end_peak_hour.time()
    # rolling_window.plot.bar(x='time_',y='travelTime',rot=90)
    # plt.tight_layout()
    # plt.savefig(f'peakHour/{each_day}.png')
    # plt.close()
    return peak_period, value


def peak_hour():
    dict_peak_hour = {}
    for each_sheet in [xl.sheet_names[i] for i in [2, 3, 4, 15, 16, 17, 22, 24, 30]]:
        df = xl.parse(each_sheet, parse_dates=['time_'])[['time_', 'travelTime']]
        dict_peak_hour[each_sheet] = plot_peak_hour(each_sheet, df)
    return dict_peak_hour
    # plt.bar(x=[f'{i[0]} - {i[1][0].strftime("%H-%M")} - {i[1][1].strftime("%H-%M")}' for i in dict_peak_hour],
    #         height=[i[2] for i in dict_peak_hour])
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig('peakHour/peakHour.png')


def plot_peak_hour_based_volume_each_lane(each_day, db):

    db.loc[:,'time_'] = db['time_'].dt.floor('T')
    rolling_window = db.iloc[::-1].rolling('60T', closed='right', on='time_').mean().round(2)[::-1]
    value = rolling_window[[f'volume_{i}' for i in range(1,6)]].max()
    start_peak_hour = rolling_window.loc[rolling_window[[f'volume_{i}' for i in range(1,6)]].idxmax(), 'time_']
    end_peak_hour = start_peak_hour + np.timedelta64(5 * 11, 'm')
    peak_period = {f'{i}':(s.time(),e.time()) for i,s,e in zip(range(1,6), start_peak_hour, end_peak_hour)}
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    # rolling_window.plot.bar()
    # plt.tight_layout()
    # plt.savefig(f'peakHour/{each_day}.png')
    # plt.close()
    return peak_period


def peak_hour_based_volume_each_lane():
    dict_peak_hour = {}
    for each_sheet in [xl.sheet_names[i] for i in [2, 3, 4, 15, 16, 17, 22, 24, 30]]:
        df = xl.parse(each_sheet, parse_dates=['time_'])[['time_', *[f'volume_{i}' for i in range(1,6)]]]
        dict_peak_hour[each_sheet] = plot_peak_hour_based_volume_each_lane(each_sheet, df)
    return dict_peak_hour
    # plt.bar(x=[f'{i[0]} - {i[1][0].strftime("%H-%M")} - {i[1][1].strftime("%H-%M")}' for i in dict_peak_hour],
    #         height=[i[2] for i in dict_peak_hour])
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig('peakHour/peakHour.png')


def plot_iteration(typ, lst):
    plt.xlabel('Iterations')
    plt.ylabel(typ, rotation=0)
    plt.title(f'{P_c} {each_sheet} {each_lane}')
    plt.plot(lst)
    plt.savefig(f'holy_moly/{typ}/{P_c} {each_sheet} {each_lane}.png')
    plt.close()


if __name__ == '__main__':
    spacing_headway = 2
    vehicle_length = 4.2
    xl = pd.ExcelFile('summary_5min_period.xlsx')
    xl_speed = pd.ExcelFile('speed_.xlsx')
    dict_peak_hour = peak_hour_based_volume_each_lane()

    xl_write = pd.ExcelWriter('holy_moly/final_result.xlsx', engine='openpyxl')
    df_dist_table_headway = pd.ExcelFile('dist_headway.xlsx')
    dict_dist_table_headway = {sheet.split('_')[0]: {f'{i}':None for i in range(1,6)} for sheet in df_dist_table_headway.sheet_names if sheet.split('_')[0] in dict_peak_hour}
    for sheet in df_dist_table_headway.sheet_names:
        day_date, lane_id = sheet.split('_')
        if day_date in dict_dist_table_headway:
            df = df_dist_table_headway.parse(sheet, parse_dates=['Time'])
            if dict_peak_hour[day_date][lane_id][0].minute>=30:
                peak_H = dict_peak_hour[day_date][lane_id][0].hour + 1
            else:
                peak_H = dict_peak_hour[day_date][lane_id][0].hour
            row_peak = df[df.Time.dt.hour==peak_H]
            dict_dist_table_headway[day_date][lane_id] = {'dist':row_peak.Distribution.values[0], 'params':row_peak.Parameters.values[0]}
            # {'Beta (4)', 'Log-normal', 'Weibull (2)', 'Gamma (2)', '****'}

    for P_c in np.round(np.arange(0.1, 1, .1).tolist(), 2):
        dict_summary_table = {}
        for each_sheet in dict_peak_hour:
            df_each_sheet = xl.parse(each_sheet)

            # ML
            df_each_sheet['VC_ratio'] = df_each_sheet[['total_vol_all_lanes']].div(561.25, axis=0)
            x = df_each_sheet.loc[:, ['VC_ratio']] ** 2
            y = df_each_sheet.loc[:, ['travelTime']]
            # train & test set
            reg = AdaBoostRegressor()
            reg.fit(x.values, y.values.ravel())

            lst_summary_table = []
            for each_lane in range(1,6):

                start_peak, end_peak = dict_peak_hour[each_sheet][str(each_lane)]
                df_each_sheet_hour = df_each_sheet[(df_each_sheet.DateTime.dt.time >= start_peak) & (df_each_sheet.DateTime.dt.time <= end_peak)]
                N_mix = df_each_sheet_hour[f'volume_{each_lane}'].sum()

                df_speed = xl_speed.parse(f'{each_sheet}_laneId_{each_lane}', usecols=['DateTime', 'timeHeadway', 'Speed'])
                df_speed_hour = df_speed[(df_speed.DateTime.dt.time > (dt.datetime.combine(dt.date(1,1,1),start_peak) - dt.timedelta(minutes=5)).time()) &
                                         (df_speed.DateTime.dt.time <= end_peak)]
                # sns.distplot(df_speed_hour.timeHeadway, hist=True)
                # plt.savefig('real.png')
                # plt.close()
                # sns.distplot(df_speed_hour.Speed, hist=True)
                # plt.savefig('real_speed.png')
                # plt.close()


                # dist_type, params = dict_dist_table_headway[each_sheet][str(each_lane)].values()
                # if 'Log-normal' in dist_type:
                #     mu, sig = re.findall(r'\d+\.\d+', params)
                #     timeHeadway_sample = np.random.lognormal(mean=mu, sigma=sig, size=1000)
                # maxValue = df_speed_hour.timeHeadway.max()
                # lognormal_sample = np.random.lognormal(mean=0.0, sigma=1.0, size=1000)
                # timeHeadway_sample = lognormal_sample - min(lognormal_sample)
                # timeHeadway_sample = timeHeadway_sample / max(timeHeadway_sample)
                # timeHeadway_sample = timeHeadway_sample * maxValue
                #
                # elif 'Beta' in dist_type:
                #     alpha, beta, *_ = re.findall(r'\d+\.\d+', params)
                #     timeHeadway_sample = np.random.beta(a=alpha, b=beta, size=1000)
                # elif 'Weibull' in dist_type:
                #     shape, *_ = re.findall(r'\d+\.\d+', params)
                #     timeHeadway_sample = np.random.weibull(a=shape, size=1000)
                # elif 'Gamma' in dist_type:
                #     shape, scale = re.findall(r'\d+\.\d+', params)
                #     timeHeadway_sample = np.random.gamma(shape=shape, scale=scale, size=1000)
                # else:
                gmm = GMM(n_components=3)
                gmm.fit(df_speed_hour.timeHeadway.values.reshape(-1, 1))
                timeHeadway_sample = gmm.sample(n_samples=10000)[0].reshape(-1)
                # sns.distplot(timeHeadway_sample[0], hist=True)
                # plt.savefig('sample.png')
                # plt.close()

                gmm = GMM(n_components=3)
                gmm.fit(df_speed_hour.Speed.values.reshape(-1, 1))
                speed_sample = gmm.sample(len(df_speed_hour))[0].reshape(-1)
                # sns.distplot(speed_sample[0], hist=True)
                # plt.savefig('sample_speed.png')
                # plt.close()

                timeHeadway_sample_cc = np.random.normal(1.2,.15,size=10000)
                N_c = round(P_c * N_mix)
                P_h = 1 - P_c
                N_h = round(P_h * N_mix)
                lst_N_mix = ['c']*N_c + ['h']*N_h
                np.random.shuffle(lst_N_mix)
                df_car_queue = pd.DataFrame({'queue':lst_N_mix})
                df_car_queue['queue_shift_neg_1'] = df_car_queue.shift(-1)
                n_cc = sum((df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='c'))
                n_hh = sum((df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='h'))
                n_hc = sum((df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='c'))
                n_ch = sum((df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='h'))
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
                h_cc_sample = np.random.choice(timeHeadway_sample_cc, n_cc, replace=False)
                df_car_queue.loc[(df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='c'),'h_mn'] = h_cc_sample  #cc
                h_ch_sample = np.random.choice(timeHeadway_sample, n_ch, replace=False)
                df_car_queue.loc[(df_car_queue.queue=='c') & (df_car_queue.queue_shift_neg_1=='h'),'h_mn'] = h_ch_sample  #ch
                h_hc_sample = np.random.choice(timeHeadway_sample, n_hc, replace=False)
                df_car_queue.loc[(df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='c'),'h_mn'] = h_hc_sample  #hc
                h_hh_sample = np.random.choice(timeHeadway_sample, n_hh, replace=False)
                df_car_queue.loc[(df_car_queue.queue=='h') & (df_car_queue.queue_shift_neg_1=='h'),'h_mn'] = h_hh_sample  #hh

                k_mixed_critical = 1 / (100 * P_c * (df_car_queue.P_mn*df_car_queue.h_mn).sum() + spacing_headway * (1 - P_c) + vehicle_length)
                k_mixed_jam = 1 / (spacing_headway * (1 - P_c) + vehicle_length)
                k = 1 / (np.random.choice(speed_sample, replace=False) * (P_c * (df_car_queue.P_mn*df_car_queue.h_mn).sum()) + spacing_headway * P_h + vehicle_length)

                if k_mixed_jam > k > k_mixed_critical:
                    dict_t, dict_c, dict_v_c = {}, {}, {}
                    lst_t_results, lst_c_results, lst_v_c_results = [], [], []
                    sum_t_predicted, sum_c_predicted, sum_v_c_predicted,  = 0, 0, 0
                    for iterate in range(40_000):
                        h_ch, h_hc, h_hh = np.random.choice(timeHeadway_sample, size=3, replace=False)
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
                    lst_summary_table.append([lst_t_results[-1], lst_c_results[-1], lst_v_c_results[-1]])
                else:
                    print(f'{P_c} {each_sheet} {each_lane}', k_mixed_critical, k, k_mixed_jam)
                    lst_summary_table.append([None,None,None])

            dict_summary_table[each_sheet] = pd.DataFrame(lst_summary_table, columns=['t','c','v_c'], index=range(1,6))
        pd.concat(dict_summary_table, axis=1).to_excel(xl_write, sheet_name=f'{P_c}')
    xl_write.save()