
# coding: utf-8

# In[11]:


def OEpredict():
   #OE COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.OE.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.OE.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)



# coding: utf-8

# In[5]:


def MEpredict():
   #ME COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.ME.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.ME.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def MCpredict():
   #MC COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.MC.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.MC.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def DCpredict():
   #DC COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.DC.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.DC.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def IM1predict():
   #IM1 COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.IM1.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.IM1.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def IM2predict():
   #IM2 COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.IM2.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.IM2.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def IM3predict():
   #IM3 COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.IM3.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.IM3.values
   dates = df.DATE.values
   
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def IM4predict():
   #IM4 COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.IM4.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.IM4.values
   dates = df.DATE.values
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def IM5predict():
   #IM5 COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.IM5.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.IM5.values
   dates = df.DATE.values
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


def IM6predict():
   #IM6 COTTON PREDICTION
   
   import pandas as pd
   import numpy as np
   location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
   df = pd.read_csv(location)
   X =[df.IM6.values][0]
   X= np.asarray(X)
   X=X.reshape(len(X),-1)
   y= df.IM6.values
   dates = df.DATE.values
   
   import numpy as np

   X[np.isnan(X)] = np.median(X[~np.isnan(X)])
   y[np.isnan(y)] = np.median(y[~np.isnan(y)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
           learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
       warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(X, y)
   res = nn.predict(X)
       #three year prediction
   predX =[res][0]
   predX= np.asarray(predX)
   predX=predX.reshape(len(predX),-1)
   predy= res
   
#dates = df.YEAR.values
   
   
   import numpy as np
   
   predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
   predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
   from sklearn.neural_network import MLPRegressor
   nn = MLPRegressor(
       hidden_layer_sizes=(10,),  activation='relu', 
   solver='adam', alpha=0.001, batch_size='auto',
       learning_rate='constant', learning_rate_init=0.01, 
   power_t=0.5, max_iter=1000, shuffle=True,
       random_state=9, tol=0.0001, verbose=False, 
   warm_start=False, momentum=0.9, nesterovs_momentum=True,
       early_stopping=False, validation_fraction=0.1, 
   beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   n = nn.fit(predX, predy)
   final =nn.predict(predX)
   f= final[-3:]
   for i in range(0,3):
       f[i]=int(f[i])
   return(f)


# In[12]:


def feedback_OE(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[1] = predict[0]
        if row[0] == date[1]:
            row[1] = predict[1]
        if row[0] == date[2]:
            row[1] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)

def feedback_ME(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[2] = predict[0]
        if row[0] == date[1]:
            row[2] = predict[1]
        if row[0] == date[2]:
            row[2] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
def feedback_MC(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[3] = predict[0]
        if row[0] == date[1]:
            row[3] = predict[1]
        if row[0] == date[2]:
            row[3] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
        
def feedback_IM6(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[10] = predict[0]
        if row[0] == date[1]:
            row[10] = predict[1]
        if row[0] == date[2]:
            row[10] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
        
def feedback_IM5(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[9] = predict[0]
        if row[0] == date[1]:
            row[9] = predict[1]
        if row[0] == date[2]:
            row[9] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
        
def feedback_IM4(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[8] = predict[0]
        if row[0] == date[1]:
            row[8] = predict[1]
        if row[0] == date[2]:
            row[8] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
        
def feedback_IM3(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[7] = predict[0]
        if row[0] == date[1]:
            row[7] = predict[1]
        if row[0] == date[2]:
            row[7] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
        
def feedback_IM2(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[6] = predict[0]
        if row[0] == date[1]:
            row[6] = predict[1]
        if row[0] == date[2]:
            row[6] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)

    
def feedback_IM1(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[5] = predict[0]
        if row[0] == date[1]:
            row[5] = predict[1]
        if row[0] == date[2]:
            row[5] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)
        
def feedback_DC(marketing):#argument: Array of size 3
    import pandas as pd
    import csv
    import numpy as np
    location = r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv'
    df = pd.read_csv(location)
    date = []
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
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
    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', mode='r') as ifile:
        allDR = list(csv.reader(ifile))
    for row in allDR:
        if row[0]==date[0]:
            row[4] = predict[0]
        if row[0] == date[1]:
            row[4] = predict[1]
        if row[0] == date[2]:
            row[4] = predict[2]

    with open(r'C:\Users\Harish\Desktop\cotton_procurement\monthwise.csv', 'w',newline = '') as ofile:
        csv.writer(ofile).writerows(allDR)



# In[19]:



from tkinter import *
from tkinter import messagebox
import calendar
import time

window = Tk()

def predict():
    global flag_label
    if(flag_label == 1):
        global label_consumption
        label_consumption.place_forget()
    type = cotton_type.get()
    if(type=="OE"):
        consumption = OEpredict()
    elif (type=="ME"):
        consumption = MEpredict()
    elif (type=="MC"):
        consumption = MCpredict()
    elif (type=="DC"):
        consumption = DCpredict()
    elif (type=="IM1"):
        consumption = IM1predict()
    elif (type=="IM2"):
        consumption = IM2predict()
    elif (type=="IM3"):
        consumption = IM3predict()
    elif (type=="IM4"):
        consumption = IM4predict()
    elif (type=="IM5"):
        consumption = IM5predict()
    elif (type=="IM6"):
        consumption = IM6predict()
    if(type=="OE"):
        consumption =[41,41,41]
    global flag
    flag = 0
    value_m1.set(consumption[0])
    value_m2.set(consumption[1])
    value_m3.set(consumption[2])
    global date
    date = time.localtime()
    global label_m1
    label_m1= Label(text=calendar.month_name[(date.tm_mon)+1][0:3])
    label_m1.place(x=50,y=225)
    
    global text_m1
    text_m1 = Entry(width=5,textvariable=value_m1)
    text_m1.place(x=50,y=250)
    
    global label_m2
    label_m2 = Label(text=calendar.month_name[(date.tm_mon)+2][0:3])
    label_m2.place(x=100,y=225)
    
    global text_m2
    text_m2 = Entry(width=5,textvariable=value_m2)
    text_m2.place(x=100,y=250)
    global label_m3
    label_m3 = Label(text=calendar.month_name[(date.tm_mon)+3][0:3])
    label_m3.place(x=150,y=225)
    global text_m3
    text_m3 = Entry(width=5,textvariable=value_m3)
    text_m3.place(x=150,y=250)
    global button_accept
    button_accept = Button(text="ACCEPT", command=lambda c=consumption: accept(c))
    button_accept.place(x=75,y=300)
    global button_reject
    button_reject = Button(text="REJECT", command=lambda c=consumption: reject(c))
    button_reject.place(x=200,y=300)
    
def reject(consumption):
    global flag
    flag = 1
    global flag_label
    flag_label = 1
    global label_consumption
    label_consumption= Label(text="Enter Marketing team forecast")
    label_consumption.place(x=75,y=200)
    del consumption[:]
    value_m1.set("")
    value_m2.set("")
    value_m3.set("")

def accept(consumption):
    f = 0
    if(flag == 1):
        if(re.match(r'^\d+$',value_m1.get(),0) and re.match(r'^\d+$',value_m2.get(),0) and re.match(r'^\d+$',value_m3.get(),0)):
            consumption.append(int(value_m1.get()))
            consumption.append(int(value_m2.get()))
            consumption.append(int(value_m3.get()))
        else:
            messagebox.showinfo(title='Invalid',message='Value is invalid(require type int)')
            f = 1
    if(f == 0):
        global date
        accept_message = "Value entered is \n" + calendar.month_name[(date.tm_mon)+1][0:3] + " : " + str(consumption[0]) + "\n" + calendar.month_name[(date.tm_mon)+2][0:3] + " : " + str(consumption[1]) + "\n" + calendar.month_name[(date.tm_mon)+3][0:3] + " : " + str(consumption[2]) 
        messagebox.showinfo(title='Accepted',message=accept_message)
    type = cotton_type.get()
    if(type=="OE"):
        feedback_OE(consumption)
    elif (type=="ME"):
        feedback_ME(consumption)
    elif (type=="MC"):
        feedback_MC(consumption)
    elif (type=="DC"):
        feedback_DC(consumption)
    elif (type=="IM1"):
        feedback_IM1(consumption)
    elif (type=="IM2"):
        feedback_IM2(consumption)
    elif (type=="IM3"):
        feedback_IM3(consumption)
    elif (type=="IM4"):
        feedback_IM4(consumption)
    elif (type=="IM5"):
        feedback_IM5(consumption)
    elif (type=="IM6"):
        feedback_IM6(consumption)
        global label_m1
        label_m1.place_forget()
        global label_m2
        label_m2.place_forget()
        global label_m3
        label_m3.place_forget()
        global text_m1
        text_m1.place_forget()
        global text_m2
        text_m2.place_forget()
        global text_m3
        text_m3.place_forget()
        global button_accept
        button_accept.place_forget()
        global button_reject
        button_reject.place_forget()
        global flag_label
        if(flag_label == 1):
            global label_consumption
            label_consumption.place_forget()
    else:
         messagebox.showinfo(title='Invalid',message='Marketing team forecast cannot be entered')       
    
flag = 0
flag_label = 0
value_m1 = StringVar()
value_m2 = StringVar()
value_m3 = StringVar()
window.title("Title")
window.geometry("500x500")
label_title = Label(text="Cotton Type").place(x=50,y=50)
cotton_type=StringVar(window)
cotton_type.set("Select")
choices = {"OE","OC","MC","DC","IM1","IM2","IM3","IM4","IM5","IM6"}
drop_down = OptionMenu(window,cotton_type,*choices).place(x=50,y=70)
button_predict =  Button(text="PREDICT",command = predict).place(x=250,y=70)
window.mainloop()

