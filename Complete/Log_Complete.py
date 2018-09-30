
# coding: utf-8

# In[9]:


def log(array):#array is an array of [date,OE,ME,,MC,DC,IM1,IM2,IM3,IM4,IM5,IM6]. It returns nothing.
    import csv
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        m = row[0]
        if(row[0]!='DATE'):
            row[0] = m[3:]
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline='') as ofile:
        csv.writer(ofile).writerows(allDR)
        
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline='') as ofile:
        csv.writer(ofile).writerows(allDR)
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([k])
    
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        #print(row[0])
        m = row[0]
        if(row[0]!='DATE' and row[0]!=''):
            n = '10-'
            n = n+m
            row[0] = n
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline='') as ofile:
        csv.writer(ofile).writerows(allDR)
from tkinter import *
from tkinter import messagebox
import re
import calendar
import time
window = Tk()
def clear():
    value_oe.set("")
    value_me.set("")
    value_mc.set("")
    value_dc.set("")
    value_im1.set("")
    value_im2.set("")
    value_im3.set("")
    value_im4.set("")
    value_im5.set("")
    value_im6.set("")

def log_value():
    values = []
    values.append(current_date)
    if(re.match(r'^\d+$',value_oe.get(),0)):
        values.append(int(value_oe.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for OE is invalid')
    if(re.match(r'^\d+$',value_me.get(),0)):
        values.append(int(value_me.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for ME is invalid')
    if(re.match(r'^\d+$',value_mc.get(),0)):
        values.append(int(value_mc.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for MC is invalid')
    if(re.match(r'^\d+$',value_dc.get(),0)):
        values.append(int(value_dc.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for DC is invalid')
    if(re.match(r'^\d+$',value_im1.get(),0)):
        values.append(int(value_im1.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for IM1 is invalid')
    if(re.match(r'^\d+$',value_im2.get(),0)):
        values.append(int(value_im2.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for IM2 is invalid')
    if(re.match(r'^\d+$',value_im3.get(),0)):
        values.append(int(value_im3.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for IM3 is invalid')
    if(re.match(r'^\d+$',value_im4.get(),0)):
        values.append(int(value_im4.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for IM4 is invalid')
    if(re.match(r'^\d+$',value_im5.get(),0)):
        values.append(int(value_im5.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for IM5 is invalid')
    if(re.search(r'^\d+$',value_im6.get(),0)):
        values.append(int(value_im6.get()))
    else:
        messagebox.showinfo(title='Invalid',message='Value for IM6 is invalid')
    print(values)    
    log(values) # values contains the value for all cotton types. Import your function and call it here.
    
window.title("Cotton Procurement Planner")
window.geometry("500x500")

value_oe = StringVar()
value_me = StringVar()
value_mc = StringVar()
value_dc = StringVar()
value_im1 = StringVar()
value_im2 = StringVar()
value_im3 = StringVar()
value_im4 = StringVar()
value_im5 = StringVar()
value_im6 = StringVar()

date = time.localtime()
#current_date = str(date.tm_mon) + "/" + str(date.tm_mday) + "/" + str(date.tm_year)
current_date = calendar.month_name[date.tm_mon][0:3] + "-" + str(date.tm_year)[2:4]
#current_date = "Feb-18"
label_date = Label(text=current_date).pack()

label_oe = Label(text="OE").place(x=50,y=50)
text_oe = Entry(width=10,textvariable=value_oe).place(x=100,y=50)

label_me = Label(text="ME").place(x=50,y=100)
text_me = Entry(width=10,textvariable=value_me).place(x=100,y=100)

label_mc = Label(text="MC").place(x=50,y=150)
text_mc = Entry(width=10,textvariable=value_mc).place(x=100,y=150)

label_dc = Label(text="DC").place(x=50,y=200)
text_dc = Entry(width=10,textvariable=value_dc).place(x=100,y=200)

label_im1 = Label(text="IM1").place(x=50,y=250)
text_im1 = Entry(width=10,textvariable=value_im1).place(x=100,y=250)

label_im2 = Label(text="IM2").place(x=250,y=50)
text_im2 = Entry(width=10,textvariable=value_im2).place(x=300,y=50)

label_im3 = Label(text="IM3").place(x=250,y=100)
text_im3 = Entry(width=10,textvariable=value_im3).place(x=300,y=100)

label_im4 = Label(text="IM4").place(x=250,y=150)
text_im4 = Entry(width=10,textvariable=value_im4).place(x=300,y=150)

label_im5 = Label(text="IM5").place(x=250,y=200)
text_im5 = Entry(width=10,textvariable=value_im5).place(x=300,y=200)

label_im6 = Label(text="IM6").place(x=250,y=250)
text_im6 = Entry(width=10,textvariable=value_im6).place(x=300,y=250)

button_clear = Button(text="CLEAR",command = clear).place(x=275,y=300)
button_clear = Button(text="LOG",command = log_value).place(x=175,y=300)

window.mainloop()

