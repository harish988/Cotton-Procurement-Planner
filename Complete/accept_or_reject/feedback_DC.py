
# coding: utf-8

# In[45]:


def feedback_DC(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
        for row in allDR[-3:]:
            if(row[0] != ''):
                date.append(row[0])
    temp1 = 0 
    for temp in df.DATE:
        temp1 = temp1+1
        if temp==date[0]:
            i = temp1
    temp1 = 0 
    for temp in df.DATE:
        temp1 = temp1+1
        if temp == date[1]:
            j = temp1
    temp1 = 0 
    for temp in df.DATE:
        temp1 = temp1+1
        if temp == date[2]:
            k = temp1
    marketing = np.array(marketing)
    predict = marketing
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[4] = predict[0]
        if row[0] == date[1]:
            row[4] = predict[1]
        if row[0] == date[2]:
            row[4] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)



