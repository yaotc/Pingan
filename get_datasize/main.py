# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import time

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]


def process():
    import pandas as pd
    import numpy as np
    
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
#build model:::
    print('building model..')
    from xgboost import XGBRegressor
    train_data = pd.read_csv(path_train)
    #train_data.head()
    print('row/col of train :',train_data.shape)
    test_data = pd.read_csv(path_test)
    #test_data.head()
    print('row/col of test  :',test_data.shape)
    input_label =[ "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED","CALLSTATE"]
    output_label = ['Y']
    train_input = train_data[input_label].values
    train_output = train_data[output_label].values
    test_input = test_data[input_label].values
    #test_output = test_data[output_label].values
    print(test_data.head())

    reg = XGBRegressor(max_depth=5, learning_rate=0.01, 
                       n_estimators=10, silent=True)
    reg.fit(train_input, train_output)
    y_pred = reg.predict(test_input)
    
    '''
    flag = 1
    for j in range(len(test_input)-1):
        if flag == 1:
            arr.append(test_data.loc[j]["TERMINALNO"])
    #for i in range(1000):
        if(test_data.loc[j+1]["TERMINALNO"] == test_data.loc[j]["TERMINALNO"]):
            j = j +1
            flag = 0
        else:
            flag = 1
    print(arr)
    print (len(arr))
    '''

    with (open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        mean = 0.0 
        k = 0
        sum = 0.0
        count = 0
        for i in range(len(test_input)-1):
            if(test_data.loc[i+1]["TERMINALNO"] == test_data.loc[i]["TERMINALNO"]):
                sum += y_pred[i]
                count += 1
            if(test_data.loc[i+1]["TERMINALNO"] != test_data.loc[i]["TERMINALNO"] or i == len(test_input)-2):
                mean = sum / (count+1)
                #print(mean,count)
                writer.writerow([test_data.loc[i]["TERMINALNO"], mean])
                k += 1
                count = 0
                sum = 0.0
                mean -= mean
                
    
    #mse
    '''
    mse = 0
    for ele in range(len(y_pred)):
        mse += (y_pred[ele] - test_output[ele][0])**2
    mse = mse / len(y_pred)
    print("mse : ", mse)
    '''
    '''
    with (open(os.path.join(path_test_out, "temp.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        for ele in range(len(y_pred)):
            mse += (y_pred[ele] - test_output[ele][0])**2
            
            writer.writerow([y_pred[ele], test_output[ele][0]])
        mse = mse / len(y_pred)
        print("mse : ", mse)
    '''



if __name__ == "__main__":
    # 程序入口
    start = time.time()
    process()
    end = time.time()
    time_elapsed = end - start
    print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))