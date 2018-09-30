
# coding: utf-8

# In[79]:



# coding: utf-8

# In[55]:


def log(array):#array is an array of [date,OE,ME,,MC,DC,IM1,IM2,IM3,IM4,IM5,IM6]. It returns nothing.
    import csv
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        m = row[0]
        if(row[0]!='DATE'):
            row[0] = m[3:]
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', 'w',newline='') as ofile:
        csv.writer(ofile).writerows(allDR)
        
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==array[0]:
            row[:] = array
            
    from dateutil.relativedelta import relativedelta
    import datetime
    today = datetime.date.today()
    #print(today)
    addMonths = relativedelta(months=3)
    future = today + addMonths
    month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    datetime.datetime.strptime(future.strftime('%m/%d/%Y'), "%m/%d/%Y")
    future = future.strftime('%m/%d/%Y')
    mo = future[0]+future[1]
    mo = int(mo)
    k = month[mo-1]
    k = k+'-'+future[8]+future[9]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', 'w',newline='') as ofile:
        csv.writer(ofile).writerows(allDR)
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([k])
    
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        #print(row[0])
        m = row[0]
        if(row[0]!='DATE' and row[0]!=''):
            n = '10-'
            n = n+m
            row[0] = n
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\feedback.csv', 'w',newline='') as ofile:
        csv.writer(ofile).writerows(allDR)

