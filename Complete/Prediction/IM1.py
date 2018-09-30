
# coding: utf-8

# In[11]:


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



