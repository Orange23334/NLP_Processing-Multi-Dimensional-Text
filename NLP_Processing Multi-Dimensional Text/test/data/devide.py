# code=utf-8
import pandas as pd
import csv

        
# 打开文件
data = pd.read_csv('./all.csv',low_memory=False)

# train
f2=open('./train.csv','w',newline='')
writer=csv.writer(f2)
with open('./all.csv','r',newline='') as f1:
    reader=csv.reader(f1)
    a=0
    for row in reader:
        writer.writerow(row)
        a=a+1
        if a==60001:
            break

# validation
save_data = data.iloc[60000: 62813]
file_name2 = 'validation.csv'  
save_data.to_csv(file_name2,index=False)  
# test
save_data = data.iloc[62814 : 65628]
file_name3 = 'test.csv'  
save_data.to_csv(file_name3,index=False)




