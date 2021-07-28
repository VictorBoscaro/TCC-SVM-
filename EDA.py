#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import defaultdict
from scipy.stats import levene
from statsmodels.tsa.stattools import grangercausalitytests

plt.style.use('dark_background')


# #### Funções

# In[2]:


def add_labels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]/2, round(y[i], 2), color = 'black', ha = 'center')


# ## Análise univariada

# ### Leitura dos dados

# #### Dados intradiários 2020-2021

# In[6]:


data = pd.read_csv(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\DataSet\btc_usd_2020.csv', header = 1)
data.columns = map(lambda x: x.casefold(), data.columns)


# In[7]:


data['date'] = pd.to_datetime(data.date)
data.set_index('date', inplace = True)
data_20 = data[data.index.year == 2020]


# #### Dados diários 2015-2020

# In[64]:


daily = pd.read_csv(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\btc_daily.csv', header = 1)
daily.columns = map(lambda x: x.casefold(), daily.columns)


# In[65]:


daily['date'] = pd.to_datetime(daily.date)
daily.set_index('date', inplace = True)
daily = daily.sort_index()
daily['diferenca'] = daily.close.diff()


# In[66]:


price_by_day = data_20.close.resample('D').apply(lambda ser: ser.iloc[-1,])
volume_by_day = data_20.volume.resample('D').sum()
resampled_by_day = pd.merge(left = price_by_day, right = volume_by_day, left_index = True, right_index = True)


# In[6]:


resampled_by_day.index.freq = 'D'


# ### Decompondo a time series agregada por dia
# <br>
# 
# 1 - Resíduos 
# - Não apresenta homocedasticidade (??)
# 
# <br>
# 
# ##### To-Do
# 
# - Ver como interpretar essas decomposições

# In[70]:


decompose = seasonal_decompose(daily.close)


# In[78]:


np.log(daily.close).plot(figsize=(20,4))
plt.xlabel('Ano')
plt.ylabel('Log do preço')
plt.title('Log do valor do Bitcoin por dia')
plt.show()


# In[71]:


plt.rcParams["figure.figsize"] = (20,10)
decompose.plot()
plt.show()


# In[90]:


plt.rcParams["figure.figsize"] = (20,5)
decompose2 = seasonal_decompose(resampled_by_day.close)
decompose2.seasonal.plot()
plt.xlabel('Ano')
plt.title('Decomposição sasonal para o ano de 2020')
plt.show()


# ### Verificando a distribuição dos retornos

# In[100]:


daily['month'] = daily.index.month


# In[102]:


sns.boxplot(data = daily, x = 'month', y = 'diferenca')


# In[141]:


grouped_mes = daily.groupby('month').diferenca.mean()
grouped_mes.plot(kind='bar')
add_labels(grouped_mes.index, grouped_mes.values)
plt.xticks(rotation=360)
plt.xlabel('Mês')
plt.ylabel('Média de Variação no Preço')
plt.title('Média de variação no Preço do Bitcoin por Mês')
plt.show()


# In[108]:


daily['dia'] = daily.index.day
sns.boxplot(data = daily, x = 'dia', y = 'diferenca')
plt.show()


# In[122]:


daily['dia_semana'] = daily.index.weekday
daily_grouped_weekday = daily.groupby(['dia_semana']).diferenca.mean()
daily_grouped_weekday.plot(kind='bar', figsize=(20, 5))
plt.xticks(rotation = 360)
plt.xlabel('Dia da Semana')
plt.ylabel('Média de Variação')
plt.title('Média na variação de preço por dia da semana')
add_labels(daily_grouped_weekday.index, daily_grouped_weekday.values)
plt.show()


# In[124]:


grouped_dia_mes = daily.groupby(['dia']).diferenca.mean()
grouped_dia_mes.plot(kind='bar', figsize=(20,5))
plt.xticks(rotation=360)
plt.show()


# In[98]:


sns.boxplot(data = daily, x = 'diferenca')


# In[96]:


diferenca_2020 = daily.dropna().diferenca.values
plt.hist(diferenca_2020)
plt.title('Distribuição para a variação diária')
plt.ylabel('Frequência')
plt.xlabel('Variação')
plt.show()


# ### Verificando variação no preço por período do dia

# In[9]:


bins = [-1, 5, 11, 17, 23]
labels = ['madrugada', 'manha', 'tarde', 'noite']
data_20.loc[:, 'periodo'] = pd.cut(data_20.index.hour, bins = bins, labels = labels)


# In[13]:


day_of_year = set(data_20.index.dayofyear)


# In[14]:


dic = {}
not_in = []
for day in day_of_year:
    for periodo in labels:
        day_period = data_20[(data_20.periodo == periodo) & (data_20.index.dayofyear == day)]
        try:
            dia = list(set(day_period.index.date))
            dia = dia[0]
            key = f'{dia}_{periodo}'
            primeiro = day_period.close.iloc[-1]
            ultimo = day_period.close.iloc[0]
            volume = day_period.volume.sum()
            dif = ultimo - primeiro
            dic[key] = [dif, volume]
        except:
            out = f'{dia}_{periodo}'
            not_in.append(out)


# In[15]:


df_periodo = pd.DataFrame(dic)
df_periodo = df_periodo.head().T
df_periodo.reset_index(inplace = True)


# In[17]:


df_periodo['data'] = df_periodo['index'].apply(lambda x: x.split('-')[0])
df_periodo['periodo'] = df_periodo['index'].apply(lambda x: x.split('_')[-1])
df_periodo.drop('index', axis = 1, inplace = True)
df_periodo.rename({0:'variacao', 1:'volume'}, axis = 1, inplace = True)


# In[19]:


to_plot = df_periodo.groupby('periodo').variacao.mean()
to_plot.index = map(lambda x: x.capitalize(), to_plot.index)
to_plot = to_plot.loc[['Madrugada', 'Manha', 'Tarde', 'Noite']]
to_plot.plot(kind='bar', figsize = (20, 5))
plt.xlabel('Período do Dia')
plt.ylabel('Média de Variação (Dólares)')
plt.title('Média de variação no preço do bitcoin por período em 2020')
plt.xticks(rotation = 45)
add_labels(to_plot.index, to_plot.values)
plt.show()


# In[15]:


to_plot = df_periodo.groupby('periodo').volume.mean()
to_plot.index = map(lambda x: x.capitalize(), to_plot.index)
to_plot = to_plot.loc[['Madrugada', 'Manha', 'Tarde', 'Noite']]
to_plot.plot(kind='bar', figsize = (20, 5))
plt.xlabel('Período do Dia')
plt.ylabel('Média de Troca (Dólares)')
plt.title('Média de volume tradado de bitcoin por período em 2020')
plt.xticks(rotation = 45)
add_labels(to_plot.index, to_plot.values)
plt.show()


# In[98]:


plt.rcParams["figure.figsize"] = (20,5)
sns.regplot(data = data_20.dropna(), y = 'diferenca', x = 'volume')
plt.show()


# In[19]:


plt.rcParams["figure.figsize"] = (20,5)
resampled_by_day['dif'] = resampled_by_day.close.diff()
sns.regplot(data = resampled_by_day.dropna(), y = 'dif', x = 'volume')
plt.show()


# In[90]:


resampled_by_day.rolling(200).mean().dropna().close.plot()


# In[93]:


data_20.rolling(200).mean().dropna().close.plot()


# In[96]:


rolling_mean = [10, 20, 50, 100, 200]
for roll in rolling_mean:
    to_plot = resampled_by_day.rolling(roll).mean().dropna()
    to_plot.close.plot(figsize=(20, 5))
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title(f'Média móvel de {roll} dias')
    plt.show()


# In[108]:


x = data_20.close.rolling(roll).mean().dropna()
index = x.iloc[:500].index
x.head()


# In[109]:


data_20.close.loc[index]


# In[101]:


for roll in np.array(rolling_mean)*100:
    to_plot = data_20.rolling(roll).mean().dropna()
    to_plot.close.plot(figsize=(20, 5))
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title(f'Média móvel de {roll} minutos')
    plt.show()


# ### Autocorrelation
# <br>
# 
# ##### Conclusões 
# <br>
# 
# - Podemos observar que não existe autocorrelação com os retornos do preço do Bitcoin quando medidos ao dia nem quando medidos em minutos ao dia.

# #### A minuto

# In[111]:


data_20['diferenca'] = data_20.close.diff()
autcor_min = acf(data_20.sort_index().iloc[:100000].diferenca.dropna())


# In[81]:


plot_acf(data_20.sort_index().iloc[:100000].diferenca.dropna())
plt.title('Autocorrelation Plot (Minute)')
plt.show()


# In[82]:


dic = {}
for t in [100, 200, 300, 400, 500, 1000]:
    mean_autocorrelation = []
    for i in range(0, t*500, t):
        key = f'test_{t}'
        autocorr = data_20.sort_index().iloc[i:i+t].diferenca
        mean_autocorr = acf(autocorr).mean()
        mean_autocorrelation.append(mean_autocorr)
        dic[key] = mean_autocorrelation


# In[110]:


for key in dic.keys():
    lis = dic[key]
    plt.hist(lis)
    plt.title(key)
    plt.show()


# #### A dia

# In[112]:


autocor_day = acf(resampled_by_day.sort_index().iloc[:100000].dif.dropna())


# In[87]:


plot_acf(resampled_by_day.sort_index().iloc[:100000].dif.dropna())


# ### Hurst Exponent
# <br><br>
# 
# #### Definição
# <br>
# 
# - O Hurst Exponent é uma medida utilizada para verificar a memória de longo prazo de séries temporais. Memória de longo prazo relaciona a taxa de dacaimento da dependência estatística entre dois pontos. Um fenômeno é considerado possuir dependência de longo prazo se a taxa de decaimento da dependência entre dois pontos for menor do que uma taxa exponencial.
# <br><br><br>
# 
# #### Interpretação:
# <br>
# 
# - O expoente de Hurst varia de 0 a 1, e podemos interpreta-lo das seguintes maneira: 
#     - 0 - 0.5: Série anti persistente. Essas séries costuam ser mean reverting, ou seja, elas tendem a retornar para sua média. 
#     - ~0.5   : Série browniana. Também chamada de random-walk. Não há nenhuma correlação entre observações passadas e futuras.
#     - 0.5 - 1: Série persistente. Um aumento no valor tende a refletir em um aumento no curto prazo, e uma dimunição no valor tende a gerar uma queda no curto prazo.
# <br><br><br>
# 
# #### Conclusão:
# <br>
# 
# - Podemos observar que, para o ano de 2020, o preço do Bitcoin apresentou uma série persistente, com um Hurst exponent de 0.62. Ou seja, existe um poder de previsibilidade em cima dessa série.

# In[57]:


data_20 = data[data.index.year == 2020].dropna()
close = data_20.close.to_frame()
change = data_20.close.pct_change().to_frame().dropna()
volume_change = data_20.volume.diff().dropna().to_frame().dropna()


# In[58]:


hurst, c, hurst_data = compute_Hc(close.values, kind = 'price')
hurst


# ### Variação por mês

# In[142]:


year_month_var = daily.resample('M').diferenca.sum()
for year in year_month_var.index.year.unique():
    year_month_var[year_month_var.index.year == year].plot(figsize=(20, 5))
    plt.title(year)
    plt.show()


# In[47]:


daily.resample('M').close.apply(lambda x: x[-1]).plot(figsize=(20, 5))


# ## Análise bivariada

# ### Granger Causality Test

# In[155]:


eth = pd.read_csv(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\DataSet\eth_2020.csv', header = 1)
eth.columns = map(lambda x: x.casefold(), eth.columns)
eth['date'] = pd.to_datetime(eth.date)
eth.set_index('date', inplace = True)
eth['diferenca'] = eth.close.diff()


# In[157]:


btc_20 = data_20.diferenca.dropna()
eth_20 = eth.diferenca.dropna()

granger_data = pd.merge(left = btc_20, right = eth_20, left_index = True, right_index = True)


# In[159]:


granger_data.columns = ['bitcoin', 'etherum']


# In[160]:


granger_data.head()


# In[163]:


grangercausalitytests(granger_data, maxlag = 2)

