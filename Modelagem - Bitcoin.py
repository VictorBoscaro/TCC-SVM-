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
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import auc, precision_recall_curve, log_loss,confusion_matrix, recall_score, precision_score, accuracy_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.datasets.loaders import load_nfl

plt.style.use('dark_background')
#plt.style.use('ggplot')


# #### Funções

# In[25]:


def split_data(df, split, targed = 'close'):
    '''Recebe um dataframe e um int para definir quantos dados serão para treinamento e a coluna da variável alvo'''
    X_train = df.drop('close', axis = 1).iloc[:split]
    y_train = np.sign(df.close).iloc[:split]
    X_test = df.drop('close', axis = 1).iloc[split:]
    y_test = np.sign(df.close).iloc[split:]
    return X_train, y_train, X_test, y_test

def get_metrics_df(df):
    '''Recebe um dataframe e retorna as métricas de um modelo de classificação'''
    accuracy = df.accuracy.iloc[0]
    precision = df.precision.iloc[0]
    recall = df.recall.iloc[0]
    au_curve = df.au_curve.iloc[0]
    tn = df.tn.iloc[0]
    tp = df.tp.iloc[0]
    fp = df.fp.iloc[0]
    fn = df.fn.iloc[0]
    fpr = df.fpr.iloc[0]
    tpr = df.tpr.iloc[0]
    return accuracy, precision, recall, au_curve, tn, tp, fp, fn, fpr, tpr

def add_lags(data, coluna, lags, prefix):
    '''Recebe um dataframe, uma coluna, uma quantidade de lags e um prefixo e retorna um dataframe'''
    cols = []
    df = data.copy()
    for lag in range(1, lags+1):
        col = f'lag_{lag}_{prefix}'
        df[col] = data[coluna].shift(lag)
        cols.append(col)
    df.dropna(inplace=True)
    return df, cols

def is_outlier(df):
    '''Recebe um dataframe e indica se as observações são outlier'''
    df = df.sort_index()
    outliers = LocalOutlierFactor(novelty = True)
    outliers_model = outliers.fit(df)
    pred_outliers = outliers_model.predict(df)
    return pred_outliers

def train_test_and_save_model(clf, X_train, X_test, y_train, y_test, comment, filename):
    '''Recebe um modelo, os dados de treino e teste e um comentário e salva as métricas calculadas para o modelo num arquivo excel'''
    clf.fit(X_train, y_train)
    y_pred_sign = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_sign)
    precision = precision_score(y_test, y_pred_sign, average = 'macro')
    recall = recall_score(y_test, y_pred_sign, average = 'macro')
    metricas = [accuracy, precision, recall]

    df_temp = pd.DataFrame(metricas, index = ['accuracy', 'precision', 'recall']).T
    now = datetime.now()
    df_temp['timestamp'] = now
    columns = ','.join(X_test.columns)
    df_temp['columns'] = columns
    df_temp['comentario'] = comment
    df_temp.set_index('timestamp', inplace = True)
    df_temp['filename'] = filename
    
    df_metricas = pd.read_excel(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\Parametros Modelos\metricas_modelos.xlsx', index_col = 0)
    df_metricas = pd.concat([df_metricas, df_temp], axis = 0)
    df_metricas.to_excel(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\Parametros Modelos\metricas_modelos.xlsx')
    del(df_metricas)
    
    return clf, df_temp, y_pred_sign

def create_hurst_vars(dataframe, hurst_samples):
    '''Recebe um dataframe e um int. O dataframe será usado para calcular o Indíce de Hurst, e o int será o número de amostras utilizadas para calcula-lo'''
    hursts = {}
    dataframe = dataframe.sort_index()
    for i in range(hurst_samples, len(dataframe)):
        try:
            filtered = dataframe.close.iloc[i-hurst_samples:i]
            idx = filtered.index[-1]
            hurst = compute_Hc(filtered.values)[0]
            hursts[idx] = hurst
        except:
            hursts[idx] = hursts[idx - timedelta(minutes=1)]
    hurst_exponent = pd.DataFrame(hursts, index = ['hurst']).T
    return hurst_exponent

def plot_validation_curve(df, target, model, cv_split = 3, score = 'accuracy', title = 'Learning Curve', train_size = np.linspace(.5, 1.0, 6)):
    '''Recebe um dataframe, a variável alvo, um modelo e um score e retorna a curva de validação'''
    X = df.drop(target, axis = 1)
    y = df[[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
    
    train_size=train_size
    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator=model, X=X_train, y=y_train, cv=cv_split, scoring=score, n_jobs=-1,
                       train_sizes=train_size,
                       return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plt.figure(figsize=(20, 5))

    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(score.capitalize())
    plt.grid()
    #plt.ylim((0,1))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


# In[30]:


data = pd.read_csv(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\DataSet\btc_daily.csv', header = 1)


# In[35]:


data.head()


# In[4]:


data.columns = map(lambda x: x.casefold().replace(' ', '_'), data.columns)
data['date'] = pd.to_datetime(data.date)
data.set_index('date', inplace = True)


# In[ ]:


hurst_index = create_hurst_vars(data, 300)


# ### Variáveis importantes

# In[6]:


data_20 = data[data.index.year == 2020].dropna()
close = data_20.close.to_frame()
change = data_20.close.pct_change().to_frame().dropna()
volume_change = data_20.volume.diff().dropna().to_frame().dropna()


# In[7]:


df_price, cols_price = add_lags(change, 'close', 5, 'price')
df_volume, cols_price = add_lags(volume_change, 'volume', 5, 'volume')


# In[8]:


data = pd.merge(left = df_price, right = df_volume, left_index = True, right_index = True)
data.drop('volume', axis = 1, inplace = True)


# In[13]:


model_data = pd.merge(right = hurst_index, left = data, left_index = True, right_index = True)


# ## Adicionando Informação sobre o Indíce de Hurst

# In[15]:


outlier = model_data.drop('hurst', axis = 1)


# In[18]:


bins = [-1, 5, 11, 17, 23]
labels = ['madrugada', 'manha', 'tarde', 'noite']
model_data.loc[:, 'periodo'] = pd.cut(model_data.index.hour, bins = bins, labels = labels)

dummies = pd.get_dummies(model_data.periodo)
model_data_per_hurst = pd.merge(left = model_data, right = dummies, left_index = True, right_index = True)


# In[5]:


rf_classifier = RandomForestClassifier(n_estimators = 200, min_samples_split = 5, min_samples_leaf = 2, max_features = 'sqrt', max_depth = 10, bootstrap = True)


# In[ ]:


model_data_per_hurst.shape


# In[14]:


df.drop('periodo', axis = 1, inplace = True)


# In[15]:


X_train, y_train, X_test, y_test = split_data(df, split = 480000)


# In[24]:


ts_split = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)


# In[25]:


k_means = KMeans(n_clusters=25)
k_means.fit(outlier)
cluster_pred = k_means.predict(outlier)


# In[31]:


model_data_per_hurst['cluster'] = cluster_pred


# In[492]:


visualizer = kelbow_visualizer(k_means, df_hurst, k=(4,30))


# In[16]:


rf_classifier.fit(X_train, y_train)


# In[61]:


y_pred = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# In[106]:


pred_prob = rf_classifier.predict_proba(X_test)


# In[109]:


pred_prob = [max(x) for x in pred_prob]


# In[114]:


test = y_test.to_frame()
test['pred'] = y_pred
test['proba'] = pred_prob
test['correct'] = test.close == test.pred
test.groupby(['correct', 'pred']).proba.mean()


# In[103]:


7687/14539


# In[41]:


model_data_per_hurst.to_csv(r'C:\Users\VictorBoscaro\Desktop\Estudos - True\TCC\Dados\Treinamento\model_data_hurst_per_cluster.csv')


# In[26]:


model, df_metricas, y_pred = train_test_and_save_model(rf_classifier, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, comment = 'RandomForestClassificator, foi usado 5 valores laggados de preço e volume, a clusterização deles, o indice de Hurst e o período do dia.', filename = 'model_data_hurst_per_cluster')


# In[50]:


accuracy, precision, recall, au_curve, tn, tp, fp, fn, fpr, tpr = get_metrics_df(df_metricas)


# In[57]:


plt.rcParams["figure.figsize"] = (20,5)
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.')
plt.title('Area Under the Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[69]:


y_test


# In[115]:


dic = {}
threshs = np.linspace(0.4,0.6,22)
for thresh in threshs:
    
    thresh = round(thresh, 2)
    
    y_pred_prob_up = (rf_classifier.predict_proba(X_test)[:,-1] >= thresh)
    y_pred_same = (rf_classifier.predict_proba(X_test)[:,1] >= thresh)
    y_pred_prob_down = (rf_classifier.predict_proba(X_test)[:,0] >= thresh)
    
    test_up = y_test.to_frame()
    test_up['order'] = range(0, len(y_test))
    
    test_down = y_test.map({-1:1, 1:-1, 0:0})
    test_down = test_down.to_frame()
    test_down['order'] = range(0, len(y_test))
    
    series_up = pd.Series(y_pred_prob_up)
    series_up.name = 'up'
    series_down = pd.Series(y_pred_prob_down)
    series_down.name = 'down'
    
    series_up_df = pd.merge(left = series_up, right = test_up, left_index = True, right_on = 'order')
    series_down_df = pd.merge(left = series_down, right = test_down, left_index = True, right_on = 'order')
    
    total_values = len(y_test)
    try:
        up_pred = series_up_df.up.value_counts().loc[True]
        up_right = series_up_df[(series_up_df.up == series_up_df.close) & (series_up_df.up == True)].shape[0]
        up_right_prop = up_right/up_pred
        up_predict_percent = up_pred/total_values
    except:
        print(f'Thresh {thresh} não teve nenhuma predição de alta correta.')
    
    try:
        down_pred = series_down_df.down.value_counts().loc[True]
        down_right = series_down_df[(series_down_df.down == series_down_df.close) & (series_down_df.down == True)].shape[0]
        down_right_prop = down_right/down_pred
        down_predict_percent = down_pred/total_values
    except:
        print(f'Thresh {thresh} não teve nenhuma predição de baixa correta.')
    
    up_returns = pd.merge(left = series_up_df['up'], right = df.loc[series_up_df.index]['close'], left_index = True, right_index = True)
    up_returns.index = pd.to_datetime(up_returns.index)
    up_index = up_returns[up_returns.up == True].index
    next_minute_up = up_index + timedelta(minutes=1)
    up_returns = up_returns[up_returns.index.isin(next_minute_up)]['close'].sum()
    
    down_returns = pd.merge(left = series_down_df['down'], right = df.loc[series_down_df.index]['close'], left_index = True, right_index = True)
    down_returns.index = pd.to_datetime(down_returns.index)
    down_index = down_returns[down_returns.down == True].index
    next_minute_down = down_index + timedelta(minutes=1)
    down_returns = down_returns[down_returns.index.isin(next_minute_down)]['close'].sum()
    
    dic[thresh] = [up_pred, up_right, up_predict_percent, up_right_prop, up_returns, down_pred, down_right, down_predict_percent, down_right_prop, down_returns]


# In[133]:


x = pd.merge(left = series_up_df['up'], right = df.loc[series_up_df.index]['close'], left_index = True, right_index = True)
x.index = pd.to_datetime(x.index)
index = x[x.up == True].index
next_index = index + timedelta(minutes=1)


# In[117]:


pd.set_option('display.max_columns', None)
pd.DataFrame(dic, index = ['up_pred', 'up_right', 'up_predict_percent', 'up_right_prop', 'up_returns', 'down_pred', 'down_right', 'down_predict_percent', 'down_right_prop', 'down_returns'])


# In[18]:


importances = rf_classifier.feature_importances_


# In[19]:


pd.Series(importances, index = X_test.columns).plot(kind='bar')

