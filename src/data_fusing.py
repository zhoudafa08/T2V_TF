#__*__encoding=utf-8__*__
#Written by Feng Zhou(fengzhou@gdufe.edu.cn)
import os, sys, time, datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import csv
import warnings
import random
warnings.filterwarnings('ignore')

def read_process_data():
    alpha101_path = '../data/Alpha101' 
    alpha191_path = '../data/Alpha191' 
    news_path = '../data/news_emotion' 
    time_frequency_path = '../data/time_frequency'
    files = os.listdir(alpha101_path)
    start_date = '2018-03-18'
    end_date = '2021-03-01'
    code_list = []
    id,id2 = 0, 0
    trading_count = 710
    for file in files:
        id2 += 1
        if not os.path.isdir(file):
            #trading = pd.read_csv(f, index_col='date')
            alpha101 = pd.read_csv(open(alpha101_path + '/' + file))
            code = file[-10:-4]
            alpha101 = alpha101[((alpha101['date']>=start_date)==True) & ((alpha101['date']<=end_date)==True)]
            alpha101 = alpha101.set_index('date')
            print('code:', code)
            
            alpha191 = pd.read_csv(open(alpha191_path+'/'+'alpha191_'+code+'.csv'))
            alpha191 = alpha191[((alpha191['date']>=start_date)==True) & ((alpha191['date']<=end_date)==True)]
            alpha191 = alpha191.iloc[:,np.r_[0, 12:alpha191.shape[1]]]
            alpha191 = alpha191.set_index('date')
    
            abnormal_ratio_101 = ((alpha101 == np.inf).sum() + (alpha101 == -np.inf).sum() + alpha101.isnull().sum()) / len(alpha101) 
            abnormal_ratio_191 = ((alpha191 == np.inf).sum() + (alpha191 == -np.inf).sum() + alpha191.isnull().sum()) / len(alpha191)
            temp = (abnormal_ratio_101 > 0.03)
            for indices in temp[temp].index.tolist():
                del alpha101[indices]
            del alpha101['alpha1_084']
            del alpha101['alpha1_095']
            temp = (abnormal_ratio_191 > 0.03)
            for indices in temp[temp].index.tolist():
                del alpha191[indices]
            
            alpha = pd.concat([alpha101, alpha191], axis=1)
            alpha['open_tomorrow'] = np.roll(alpha['open'], -1)
            del alpha['code']
            del alpha['open_yesterday']
            del alpha['close_yesterday']
            
            trading = alpha.iloc[:, np.r_[0:8, -1]]
            alpha = alpha.iloc[:, 8:-1]
            
            new_trading = trading.copy() 
            for indices in trading.columns.tolist():
                where_inf = np.isinf(trading[indices])
                trading[indices][where_inf] = np.nan
                cond = trading[indices].isna()
                new_trading[indices][~cond] = preprocessing.scale(trading[indices].dropna())
                new_trading[indices][cond] = 0
            
            new_alpha = alpha.copy() 
            for indices in alpha.columns.tolist():
                if alpha[indices].dtypes != bool:
                    where_inf = np.isinf(alpha[indices])
                    alpha[indices][where_inf] = np.nan
                    cond = alpha[indices].isna()
                    new_alpha[indices][~cond] = preprocessing.scale(alpha[indices].dropna())
                    new_alpha[indices][cond] = 0
                else:
                    temp_data = pd.get_dummies(alpha[indices])
                    temp_data.columns=[indices+'_False',indices+'_True']
                    del new_alpha[indices]
                    new_alpha = new_alpha.join(temp_data)
            print('Alpha shape:', new_alpha.shape)
            
            min_column_list = np.load('../data/least_column_list.npy',allow_pickle=True).item()
            for indices in new_alpha.columns.tolist():
                if indices not in min_column_list:
                    del new_alpha[indices]
            min_column_num = len(min_column_list)
            if new_alpha.shape[1] != min_column_num:
                continue
            pca_dim=10
            pca = PCA(n_components=pca_dim)
            pca_results = pca.fit_transform(new_alpha)
            pca_results = pd.DataFrame(pca_results, columns=['alpha'+str(i+1) for i in range(pca_dim)]) 
            pca_results['date'] = new_alpha.index
            pca_results = pca_results.set_index('date')
            
            emotion = pd.read_csv(open(news_path+'/'+'emotion_news_'+code+'.csv'))
            emotion = emotion[((emotion['date']>=start_date)==True) & ((emotion['date']<=end_date)==True)]
            emotion = emotion.set_index('date')
            emotion.columns = ['avg_snow', 'std_snow', 'avg_senta', 'std_senta']
            new_emotion = emotion.copy()
            print(new_emotion.mean())
            for indices in emotion.columns.tolist():
                new_emotion[indices] = preprocessing.scale(emotion[indices])  
    
            time_freq_date = pd.read_csv(open(time_frequency_path+'/'+code+'_if_date.csv'))
            time_freq_feats = pd.read_csv(open(time_frequency_path+'/'+code+'_if_feats.csv'), header=None)
            time_freq = pd.concat([time_freq_date, time_freq_feats], axis=1)
            time_freq = time_freq[((time_freq['all_date']>=start_date)==True) & ((time_freq['all_date']<=end_date)==True)]
            time_freq = time_freq.set_index('all_date')
    
            new_time_freq = time_freq.copy()
            for indices in time_freq.columns.tolist():
                new_time_freq[indices] = preprocessing.scale(time_freq[indices])  
            
            time_freq_labels = pd.read_csv(open(time_frequency_path+'/'+code+'_if_labels.csv'), header=None)
            time_freq_base_close = pd.read_csv(open(time_frequency_path+'/'+code+'_if_base_close.csv'), header=None)
            label = pd.concat([time_freq_date, time_freq_labels, time_freq_base_close], axis=1)
            label = label[((label['all_date']>=start_date)==True) & ((label['all_date']<=end_date)==True)]
            label = label.set_index('all_date')
            #base_close = pd.concat([time_freq_date, time_freq_base_close], axis=1)
            #base_close = base_close[((base_close['all_date']>=start_date)==True) & ((base_close['all_date']<=end_date)==True)]
            #base_close = base_close.set_index('all_date')
    
            tmp_data = pd.concat([new_trading, pca_results, new_emotion], axis=1, join='inner')
            window_sizes = 5
            inputs_lagged = pd.DataFrame()
            init_value = tmp_data.iloc[0]
            for window_size in range(window_sizes):
                inputs_roll = np.roll(tmp_data, window_size, axis=0)
                inputs_roll[:window_size, :] = init_value
                inputs_roll = pd.DataFrame(inputs_roll, index=tmp_data.index,
                                            columns=[i + '_lag{}'.format(window_size) for i in tmp_data.columns])
                inputs_lagged = pd.concat([inputs_lagged, inputs_roll], axis=1)
            
            samples = pd.concat([inputs_lagged, new_time_freq, label], axis=1, join='inner')
            #print('start:', samples.head())
            #print('end:', samples.tail())
            #print('size:', samples.shape)
            

            feats = samples.iloc[:, :-2]
            #base_close = samples.iloc[:,-2]
            labels = samples.iloc[:,-2:]
            #print(feats.shape, feats.head())
            #print(base_close.shape, base_close.head())
            print(labels.shape, labels.head())
            
            if len(feats) == trading_count and id == 0:
                a50_feats = feats.values[..., np.newaxis]
                #a50_base_close = base_close.values.reshape(-1, 1)[..., np.newaxis]
                a50_labels = labels.values.reshape(-1, 2)[..., np.newaxis]
                code_list.append(code)
                print(code)
            elif len(feats) == trading_count:
                a50_feats = np.dstack((a50_feats, feats.values[..., np.newaxis]))
                #a50_base_close = np.dstack((a50_base_close, base_close.values.reshape(-1,1)[..., np.newaxis]))
                a50_labels = np.dstack((a50_labels, labels.values.reshape(-1,2)[..., np.newaxis]))
                code_list.append(code)
                print(code)
            id +=1
            print(id, a50_feats.shape)
    
    
    a50_feats = np.transpose(a50_feats, (0,2,1))
    a50_labels = np.transpose(a50_labels, (0,2,1))
    new_samples = np.zeros((len(a50_feats), a50_feats.shape[1], window_sizes, (18+pca_dim)))
    for i in range(18+pca_dim):
        if i <= (18+pca_dim-window_sizes-1):
            new_samples[:, :, :, i] = a50_feats[:, :, i:(18+pca_dim-window_sizes)*window_sizes:(18+pca_dim-window_sizes)]
        else:
            new_samples[:, :, :, i] = a50_feats[:, :, i*window_sizes:(i+1)*window_sizes] 
    a50_labels = a50_labels[..., np.newaxis]
    
    print(new_samples.shape, a50_labels.shape)
    return new_samples, a50_labels, code_list

if __name__== '__main__':
    x, z, code_list = read_process_data()
    with open('../data/a50_feats_pca10_wnd5.npy', 'wb') as f:
        np.save(f, x)
    with open('../data/a50_labels_pca10_wnd5.npy', 'wb') as f:
        np.save(f, z)
    with open('../data/code_list', 'w') as f:
        write = csv.writer(f)
        write.writerows(code_list)

