# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import time
import gc
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
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    train_data = pd.read_csv(path_train)
    train_data.head()

    print('row/col of train :',train_data.shape)

    #input_label =["TERMINALNO","TRIP_ID","LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED"]
    input_label =["TERMINALNO","TRIP_ID", "DIRECTION", "HEIGHT"]
    output_label = ['Y']
    train_input = train_data[input_label]    
    train_output = train_data[output_label]

    del train_data
    gc.collect()
    
    datanum = len(train_input)
    #train_input = train_input.loc[0:datanum]
    #train_output = train_output.loc[0:datanum]


    start = time.time()  

    train_input1 = train_input.drop(0)
    train_input1.loc[datanum] = train_input1.loc[1]
    train_input1 = train_input1.reset_index(drop=True)
    train_input = train_input1 - train_input
    train_input = train_input.abs()
    #train_input1.drop(train_input1.index,inplace=True)



    train_input['Y'] = train_output['Y'] 
    train_input = train_input[(train_input.TERMINALNO==0)&(train_input.TRIP_ID==0)]

    del train_input1
    del train_output
    gc.collect()

 
    end = time.time()
    time_elapsed = end -start
    print('train data are processed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    start = time.time()

    reg = XGBRegressor(max_depth=9,min_child_weigh=5,
                    eta=0.025,gamma=0.06,
                    subsample=1,learning_rate=0.01, 
                    n_estimators=50,silent=True, 
                    n_jobs=-1,objective='reg:linear')
    reg.fit(train_input[["DIRECTION", "HEIGHT"]], train_input[output_label])

    end = time.time()
    time_elapsed = end -start
    print('training data in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    train_input.drop(train_input.index,inplace=True)


    test_data = pd.read_csv(path_test)
    pre_input = test_data[input_label]

    del test_data
    gc.collect()

    start = time.time()  

    pre_input1 = pre_input.drop(0)
    pre_input1.loc[datanum] = pre_input1.loc[1]
    pre_input1 = pre_input1.reset_index(drop=True)
    pre_input = pre_input1 - pre_input
    pre_input = pre_input.abs()

    del pre_input1
    gc.collect()

    a = pre_input[(pre_input.TERMINALNO!=0)].index.tolist()#yuansu xiabiao  

    y_pred = reg.predict(pre_input[["DIRECTION", "HEIGHT"]])

    end = time.time()
    time_elapsed = end -start
    print('test data in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print (len(a))
     
    pre_input['Y']=y_pred
    with (open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for i in range(len(a)):
            if i==0:
                part = pre_input[0:a[i]]
            else:
                part = pre_input[a[i-1]:a[i]]
            avg = (part[(part.TERMINALNO==0)&(part.TRIP_ID==0)]['Y'].max()
            +part[(part.TERMINALNO==0)&(part.TRIP_ID==0)]['Y'].mean())*0.5
            writer.writerow([i+1, avg])



if __name__ == "__main__":
    print("****************** start **********************")
    # 程序 入口
    
    process()


