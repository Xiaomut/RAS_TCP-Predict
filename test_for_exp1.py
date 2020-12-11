import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')


def todatetime(data):
    if isinstance(data, str):
        pass
    else:
        data = str(data)
    return int(time.mktime(time.strptime(data, '%Y-%m-%d %H:%M:%S')))


def read_json(file):
    with open(file, 'r') as f:
        r = json.load(f)
    return r


if __name__ == "__main__":
    # filename = r'X:\notebook\AI\data1\Data_RAS\ExampleTestSet_0800.csv'
    filename = sys.argv[1]
    test_datas = pd.read_csv(filename)

    test_datas['PlannedTime'] = test_datas['PlannedTime'].apply(todatetime)
    test_datas['Realisation'] = test_datas['Realisation'].apply(todatetime)

    pre_delays = []
    departure = ''

    # 1. 处理延时时间
    extract_minines = test_datas[['Trainnumber', 'Delay_min']].values.tolist()
    temp_number = extract_minines[0][0]
    for index, extract_minine in enumerate(extract_minines):
        if temp_number != extract_minine[0]:
            # 能不用列表就少用，减少内存消耗
            #         got_number.append(extract_number[0])
            temp_number = extract_minine[0]
            pre_delay = 0
        else:
            pre_delay = -1.
            try:
                pre_delay = extract_minines[index - 1][1]
            except:
                pass
        pre_delays.append(pre_delay)
    test_datas['PreDelay_min'] = pre_delays
    test_datas['PreDelay_min'][0] = 0

    # 2. 处理计划时间得到时间差
    extract_times = test_datas[['Trainnumber', 'PlannedTime',
                                'Realisation']].values.tolist()
    pre_plantimes = []
    pre_realtimes = []

    temp_number = extract_times[0][0]
    for index, extract_time in enumerate(extract_times):
        # 如果是始发站，那么时间就赋予它原来的时间
        if temp_number != extract_minine[0]:
            #         got_number.append(extract_number[0])
            temp_number = extract_minine[0]
            pre_plantime = extract_time[1]
            pre_realtime = extract_time[2]
        # 非始发站，时间赋予上一站的时间
        else:
            pre_plantime = extract_times[index - 1][1]
            pre_realtime = extract_times[index - 1][2]
        pre_plantimes.append(pre_plantime)
        pre_realtimes.append(pre_realtime)

    test_datas['Pre_plantimes'] = pre_plantimes
    test_datas['Pre_realtimes'] = pre_realtimes

    test_datas[
        'residual'] = test_datas['PlannedTime'] - test_datas['Pre_realtimes']

    # 3. 得到距离参数
    locations_json = read_json('./locations.json')
    distances = []
    got_number = []
    departure = ''

    extract_numbers = test_datas[['Trainnumber', 'Location']].values.tolist()

    for index, extract_number in enumerate(extract_numbers):
        if extract_number[0] not in got_number:
            got_number.append(extract_number[0])
            distance = 0
        else:
            distance = -1.
            try:
                dises = locations_json[extract_numbers[index - 1][1]]
                for dis in dises:
                    if dis[0] == extract_number[1]:
                        distance = dis[1]
                        break
            except:
                pass
        distances.append(distance)
    test_datas['distance'] = distances

    # 4. 对 Activity 这一列进行 one-hot 化
    df_activity = pd.get_dummies(test_datas['Activity'])
    test_datas_final = pd.concat([test_datas, df_activity], axis=1)

    # 5. 组成特征
    features = test_datas_final[[
        'A', 'D', 'K_A', 'K_V', 'V', 'distance', 'PreDelay_min', 'residual',
        'PlannedTime'
    ]]

    # 6. 加载模型
    clf = joblib.load('./models/exp1_model.pkl')
    # 7. 预测结果

    ss_X = StandardScaler()
    features = ss_X.fit_transform(features)
    predict = clf.predict(features)
    print(predict)