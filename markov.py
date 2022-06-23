import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
from sklearn.cluster import KMeans

def is_cont(df):
    point = np.array(df['지점'])
    location = np.array(df['지점명'])
    date = np.array(df['일시'])
    snow = np.array(df['적설(cm)']) * 10
    year = []
    month = []
    day = []
    hour = []
    strp = []
    mk = []
    continuous_1_hour = []
    continuous_3_hours = []
    continuous_6_hours = []

    for n, t in enumerate(date):
        t = time.strptime(t, '%Y-%m-%d %H:%M')
        strp.append(t)
        mk.append(time.mktime(t))
        year.append(t.tm_year)
        month.append(t.tm_mon)
        day.append(t.tm_mday)
        hour.append(t.tm_hour)

    for n, k in enumerate(mk):
        is_1_hour = (k + 3600.0 in mk)
        is_3_hours = (k + 10800.0 in mk)
        is_6_hours = (k + 21600.0 in mk)
        continuous_1_hour.append(is_1_hour)
        continuous_3_hours.append(is_3_hours)
        continuous_6_hours.append(is_6_hours)
    dic = {'point': point, 'location': location, 'snow': snow, 'year': year, 'month': month, 'day': day, 'hour': hour, 'con_1h': continuous_1_hour, 'con_3h': continuous_3_hours, 'con_6h': continuous_6_hours}
    df2 = pd.DataFrame(dic)
    return df2

def make_cluster_in_order(df, n_clusters):
    snow = np.log(np.array(df['snow'])).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit(snow)
    df['cluster']=clusters.labels_
    minmax = []

    dtype = [('min', float), ('max', float), ('idx', int)]
    for i in set(df['cluster']):
        cc = df[df['cluster'] == i]
        minmax.append((cc['snow'].min(), cc['snow'].max(), i))

    # minmax = np.sort(minmax, axis=0)
    minmax = np.array(minmax, dtype=dtype)
    minmax = np.sort(minmax, axis=0, order='min')
    
    cluster_in_order = []
    for n, sn in enumerate(df['snow']):
        for k, m in enumerate(minmax):
            if m[0] <= sn <= m[1]:
                cluster_in_order.append(k)
    df['cluster'] = cluster_in_order
    
    
    return df, minmax
            
def make_cluster_test_data(df, minmax):
    cluster_test = []
    snow = (np.array(df['snow']))
    for n, sn in enumerate(df['snow']):
        for k, m in enumerate(minmax):
            if k == len(minmax) - 1:
                cluster_test.append(k)
                break
            elif m[0] <= sn < minmax[k+1][0]:
                cluster_test.append(k)
                break
    df['cluster'] = cluster_test
    
    return df
                

def markov():
    df = pd.read_csv('data/2002.csv', encoding='euc-kr')
    for y in range(2003, 2023):
        new_df = pd.read_csv('data/{}.csv'.format(y), encoding='euc-kr')
        df = pd.concat([df, new_df])
    df = df.reset_index(drop=True)

    point_idx = list(set(df['지점']))
    idx = np.array(df.groupby('지점').count()['지점명'] > 5000)
    points = point_idx * idx
    points_idx = []
    for p in points:
        if p > 0:
            points_idx.append(p)
    
    n_clusters = 20
    train_by_points_times = {}
    for key in points_idx:
        train_by_points_times[key] = []

    for hours in range(1, 3):
        print("Start at ", hours)
        for key in points_idx:
            df_90 = df[df['지점']==key].fillna(0)
            df_90 = df_90[df_90['적설(cm)']>0]
            df2_90 = is_cont(df_90)
            df2_90_train = df2_90[df2_90['year'] < 2018]
            df2_90_test = df2_90[df2_90['year'] >= 2018]
            df2_90_train, minmax = make_cluster_in_order(df2_90_train, n_clusters)
            markov = np.zeros((n_clusters, n_clusters))
            for n, cl in enumerate(df2_90_train['cluster']):
                if n == len(df2_90_train['cluster'])-6:
                    break
                markov[cl][df2_90_train['cluster'][n+1]] += 1
            markov_prob = np.zeros(markov.shape)
            for n, row in enumerate(markov):
                row /= np.sum(row)
                markov_prob[n] = row
            for i in range(hours - 1):
                markov_prob = markov_prob @ markov_prob
            markov_cum = np.zeros(markov.shape)
            for n, row in enumerate(markov_prob):
                for m in range(len(row)):
                    markov_cum[n][m] = np.sum(row[:m+1])
            df2_90_test = make_cluster_test_data(df2_90_test, minmax)
            pred_snow_ = []
            for i in range(10):
                pred_cluster = [999] * hours
                for n, cl in enumerate(df2_90_test['cluster']):
                    p = np.random.rand()
                    row = markov_cum[cl]
                    for k, r in enumerate(row):
                        if p < r:
                            pred_cluster.append(k)
                            break
                pred_cluster = pred_cluster[:-hours]
                df2_90_test['pred_cluster'] = pred_cluster
                pred_snow = [0] * hours
                for n, cl in enumerate(df2_90_test['pred_cluster'][hours:]):
                    a, b, _ = minmax[cl]
                    pred_sn = np.random.uniform(a, b, 1)[0]
                    pred_snow.append(pred_sn)
                pred_snow_.append(pred_snow)
            pred_snow_ = np.array(pred_snow_)
            mean = np.mean(pred_snow_, axis=0)
            mean_95_ci = 1.96 * np.std(pred_snow_, axis=0) / np.sqrt(10)
            # train_by_points_times[hours][key].append(mean)
            # train_by_points_times[hours][key].append(mean - mean_95_ci)
            # train_by_points_times[hours][key].append(mean + mean_95_ci)
            snow_test = df2_90_test['snow'][hours:]
            pred_snow_test = mean[hours:]
            pred_snow_low_int = (mean - mean_95_ci)[:-hours]
            pred_snow_high_int = (mean + mean_95_ci)[:-hours]
            rmse = np.sqrt(np.sum((snow_test - pred_snow_test) ** 2) / len(pred_snow_test))
            train_by_points_times[key].append(rmse)
        
    print(train_by_points_times)


if __name__ == '__main__':
    markov()