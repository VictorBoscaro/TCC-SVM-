#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
from scipy import stats
from sklearn.svm import SVC
import glob
from os import listdir
from os.path import isfile, join
from datetime import datetime
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import time
import pickle
from hurst import compute_Hc
from sklearn.neighbors import LocalOutlierFactor

plt.style.use('dark_background')
#plt.style.use('ggplot')


# In[2]:


data = pd.read_csv(r'C:\Users\VictorBoscaro\Desktop\Estudos\TCC\Dados\btc_usd_2020.csv', header = 1)


# In[3]:


data.head()


# In[4]:


data.info()


# # Resumo
# <br>
# 
# Neste notebook será encontrado uma base de dados com informações a respeito do preço do bitcoin no ano de 2020. Sua granularidade é por minuto. Sim, é dado para um caralho. E aparentemente eles estão limpos.
# <br>
# 
# O preço do bitcoin se da em dólares. 
# <br><br>
# 
# ## Um breve resumo 
# - 680300 observações
# - 0 informações nulas
# 
# <br>
# Para fins de experimentação, e para meu computador não explodir, as agragações nesse notebook serão feita num período de 3 meses. <br>
#         - Agora (21/05) que a elo me mandou um notebook bom talvez não exploda tão fácil.
# 
# <br><br>
# Esse períododo será de Janeiro de 2021 até Março de 2021. Isso nos deixa com:
# - 4320 observações
# 
# <br><br>
# 
# ## To-Do
# - Outlier Score - Procurar 

# ## Uma rápida olhada nos dados

# In[5]:


data.describe()


# In[4]:


data.columns = map(lambda x: x.casefold().replace(' ', '_'), data.columns)
data.head()


# In[5]:


data['date'] = pd.to_datetime(data.date)
data.set_index('date', inplace = True)
#data.drop('unix_timestamp', axis = 1, inplace = True)


# In[8]:


data.isnull().sum()


# In[8]:


data[data.index.year == 2020].close.plot(figsize=(15, 5))
plt.xlabel('Data')
plt.ylabel('Preço do Bitcoin (USD)')
plt.title('Variação do preço do Bitcoin, em dólares, para o ano de 2020')
plt.show()


# ## Comprar ou não

# ### Variáveis importantes

# In[9]:


data_20 = data[data.index.year == 2020].dropna()
close = data_20.close.to_frame()
change = data_20.close.pct_change().to_frame().dropna()
volume_change = data_20.volume.diff().dropna().to_frame()


# ## Transformando os dados em estacionários

# In[11]:


adfuller(close.iloc[:50000])


# In[12]:


adfuller(change.iloc[:50000])


# ## Função para adicionar os lags que serão usados como features para o modelo

# In[10]:


def add_lags(data, coluna, lags, agg):
    cols = []
    df = data.copy()
    for lag in range(1, lags+1):
        col = f'lag_{lag}_{agg}'
        df[col] = data[coluna].shift(lag)
        cols.append(col)
    df.dropna(inplace=True)
    return df, cols


# In[50]:


df_price, cols_price = add_lags(change, 'close', 5, 'price')
df_volume, cols_volume = add_lags(volume_change, 'volume', 5, 'volume')


# In[47]:


data = pd.merge(left = df_price, right = df_volume, left_index = True, right_index = True)
data.drop('volume', axis = 1, inplace = True)


# In[24]:


x_train = data.iloc[-5000:]


# In[28]:


x_test = data.iloc[:-5000]


# In[29]:


outliers = LocalOutlierFactor(novelty = True)
outliers_model = outliers.fit(x_train)


# In[30]:


predict = outliers_model.predict(x_test)


# In[31]:


x_test['outlier'] = predict


# In[38]:


model_data = x_test.iloc[:20000]
model_data


# ## Treinando um SVM

# In[56]:


model = SVC(C=100, probability = True)
split = 15000
train_x = np.sign(model_data[cols_price + cols_volume]).iloc[:split]
train_y = np.sign(model_data.close).iloc[:split]
test_x = np.sign(model_data[cols_price + cols_volume]).iloc[split:]
test_y = np.sign(model_data.close).iloc[split:]
model.fit(np.sign(train_x[cols_price + cols_volume]), np.sign(train_y))
pred = model.predict(test_x)
strat = pred*test_y


# In[60]:


test_y = test_y.to_frame()


# In[62]:


test_y['prediction'] = pred


# In[64]:


test_y[test_y.close != test_y.prediction]


# In[ ]:


test_y[test_y.close == test_y.prediction]


# In[21]:


test_set = model_data.iloc[split:]
test_set.loc[:, 'prediction'] = pred


# In[22]:


test_set.loc[:, 'real'] = np.sign(test_set.close)


# In[23]:


test_set.real.value_counts()


# In[24]:


test_set[test_set.prediction == test_set.real].shape


# ## Hurst Exponent
# <br><br>
# 
# ### Definição
# <br>
# 
# - O Hurst Exponent é uma medida utilizada para verificar a memória de longo prazo de séries temporais. Memória de longo prazo relaciona a taxa de dacaimento da dependência estatística entre dois pontos. Um fenômeno é considerado possuir dependência de longo prazo se a taxa de decaimento da dependência entre dois pontos for menor do que uma taxa exponencial.
# <br><br><br>
# 
# ### Interpretação:
# <br>
# 
# - O expoente de Hurst varia de 0 a 1, e podemos interpreta-lo das seguintes maneira: 
#     - 0 - 0.5: Série anti persistente. Essas séries costuam ser mean reverting, ou seja, elas tendem a retornar para sua média. 
#     - ~0.5   : Série browniana. Também chamada de random-walk. Não há nenhuma correlação entre observações passadas e futuras.
#     - 0.5 - 1: Série persistente. Um aumento no valor tende a refletir em um aumento no curto prazo, e uma dimunição no valor tende a gerar uma queda no curto prazo.
# <br><br><br>
# 
# ### Conclusão:
# <br>
# 
# - Podemos observar que, para o ano de 2020, o preço do Bitcoin apresentou uma série persistente, com um Hurst exponent de 0.62. Ou seja, existe um poder de previsibilidade em cima dessa série.

# In[38]:


hurst, c, hurst_data = compute_Hc(close.values, kind = 'price')


# In[39]:


hurst


# In[26]:


filename = r'C:\Users\Elogroup\OneDrive\Área de Trabalho\Estudos\TCC\svm.sav'
pickle.dump(model, open(filename, 'wb'))

