import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from scipy.stats import pearsonr
import collections
sns.set_context('paper', font_scale=2)

lgdcnn_dir = r"D:\deep\LGDCNN"
def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

def KL_divergence(p,q):
    return scipy.stats.entropy(p, q, base=2)

def eculidDisSim(x,y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))

def manhattanDisSim(x,y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a-b) for a,b in zip(x,y))

def cosSim(x,y):
    '''
    余弦相似度
    '''
    tmp=np.sum(x*y)
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return np.round(tmp/float(non),9)

def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]

# s_d = 'ENV_OQMD_band_test'
# s_d = 'ENV_aflow_band_test'
# s_d = 'ENV_OQMD_Formation_Enthalpy_test'
# s_d = 'ENV_OQMD_Energy_per_atom_test'
# s_d = 'ENV_aflow__energy_atom_test'
# s_d = 'ENV_matbench_mp_gap_test'
# s_d = 'ENV_matbench_log_gvrh0_test'
# s_d = 'ENV_OQMD_Volume_per_atom_test'

# dir_s = 'CECV/embeddings_LSTM'
# dir_s = 'CECV/embeddings_DCNN_K1'
# dir_s = 'CECV/embeddings_DCNN'
# dir_s = 'CECV/embeddings_LDCNN' #LDCNN

# s_n = 'TransformerEncoder' + str(num) + '_layers'
# s_n = 'Dropout' + str(num) + '_layers'
# s_n = 'LSTM_output' + str(num)
# s_n = 'GRU_output' + str(num)
# s_n = 'RNN_output' + str(num)
# s_n = 'embeddings_RDCNN' + str(num)


str_f = 'Si'
s_d = 'ENV_CritExam_Ed_test'
dir_s = 'CECV/embeddings_L-G-DCNN'  
num = 5
s_n = 'DCNN_sequential_' + str(num) + '_layers'


for thred_v in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
    s_name = '3'
    data_dir = os.path.join(lgdcnn_dir,"data", dir_s, s_d, str_f,s_n,s_name)
    if isinstance(data_dir, list):
        mat_props = data_dir
    elif os.path.isdir(data_dir):
        mat_props = os.listdir(data_dir)
    else:
        mat_props = [data_dir]
    name1 = [i.split('.csv')[0] for i in mat_props if i.split('.csv')[0] == 'fraction_fomula']
    path_formula = os.path.join(data_dir, f'{name1[0]}.csv')
    df_f = pd.read_csv(path_formula)
    formu = df_f['formula'].values
    row_ind = 3
    save_num = np.zeros((len(mat_props)-1,row_ind,row_ind))
    for i, mat_prop in enumerate(mat_props):
        if mat_prop.split('.csv')[0] == 'fraction_fomula':
            mat_props.pop(i)
    for i, mat_prop in enumerate(mat_props):   
        if os.path.isfile(mat_prop):
            name = os.path.basename(mat_prop).split('.csv')[0]
            path = mat_prop
        elif os.path.isdir(data_dir):
            name = mat_prop.split('.csv')[0]
            path = os.path.join(data_dir, f'{name}.csv')
        else:
            path = data_dir
            name = mat_prop.split('/')[-1].split('.csv')[0]
            
        # read csv files
        df = pd.read_csv(path)
        if 'element' not in df.columns:
            df['element'] = df.iloc[:,0]
            df = df.drop(columns=['Unnamed: 0'])

        # get rid of 'Null' element representation
        null_row = df['element'] == 'Null'
        df = df[~null_row]

        # grab only the elements that are in oliynyk
        df.index = df['element']
        elements = df.pop('element')
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        current_elements = elements
        df_extended = df.reindex(current_elements)
        dfT = df_extended.T
        dfT.columns = current_elements
        corr2 = dfT.corr()
        save_num[i,:,:] = corr2.values
    all_dict = dict(zip(formu, save_num)) 
    index_coll = collections.OrderedDict()
    k = 0
    for k1,values in all_dict.items():
        ll_k = []
        ll_v = []
        vv_k = []
        vv_v = []
        for k2,value_q in all_dict.items():
            if np.abs((np.subtract(np.abs(values), np.abs(value_q)))).max() < thred_v:
                ll_v.append(value_q)
                ll_k.append(k2)
            else:
                vv_v.append(value_q)
                vv_k.append(k2)
        if len(ll_v) != 0:
            index_coll[str(k)] = dict(zip(ll_k, ll_v))
            k = k+1
        if len(vv_v) != 0 and len(vv_v) != 1: 
            all_dict = dict()
            all_dict = dict(zip(vv_k, vv_v))
        else:
            break
    if len(vv_v) != 0:       
        index_coll[str(k)] = dict(zip(vv_k, vv_v))

#%%
    s_name = '4'
    data_dir = os.path.join(lgdcnn_dir,"data", dir_s, s_d, str_f,s_n,s_name)
    if isinstance(data_dir, list):
        mat_props = data_dir
    elif os.path.isdir(data_dir):
        mat_props = os.listdir(data_dir)
    else:
        mat_props = [data_dir]
    name1 = [i.split('.csv')[0] for i in mat_props if i.split('.csv')[0] == 'fraction_fomula']
    path_formula = os.path.join(data_dir, f'{name1[0]}.csv')
    df_f = pd.read_csv(path_formula)
    formu_4 = df_f['formula'].values
    row_ind_4 = 4
    save_num_4 = np.zeros((len(mat_props)-1, row_ind_4, row_ind_4))
    for i, mat_prop in enumerate(mat_props):
        if mat_prop.split('.csv')[0] == 'fraction_fomula':
            mat_props.pop(i)
    for i, mat_prop in enumerate(mat_props):   
        if os.path.isfile(mat_prop):
            name = os.path.basename(mat_prop).split('.csv')[0]
            path = mat_prop
        elif os.path.isdir(data_dir):
            name = mat_prop.split('.csv')[0]
            path = os.path.join(data_dir, f'{name}.csv')
        else:
            path = data_dir
            name = mat_prop.split('/')[-1].split('.csv')[0]
        # read csv files
        df = pd.read_csv(path)
        if 'element' not in df.columns:
            df['element'] = df.iloc[:,0]
            df = df.drop(columns=['Unnamed: 0'])
        # get rid of 'Null' element representation
        null_row = df['element'] == 'Null'
        df = df[~null_row]
        # grab only the elements that are in oliynyk
        df.index = df['element']
        elements = df.pop('element')
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        current_elements = elements
        df_extended = df.reindex(current_elements)
        dfT = df_extended.T
        dfT.columns = current_elements
        corr2 = dfT.corr()
        save_num_4[i,:,:] = corr2.values
    all_dict_4 = dict(zip(formu_4, save_num_4)) 
    index_coll_4 = collections.OrderedDict()
    k = 0
    for k1,values in all_dict_4.items():
        ll_k = []
        ll_v = []
        vv_k = []
        vv_v = []
        for k2,value_q in all_dict_4.items():
            if np.abs(np.subtract(values, value_q)).max() <= thred_v:
                ll_v.append(value_q)
                ll_k.append(k2)
            else:
                vv_v.append(value_q)
                vv_k.append(k2)
        if len(ll_v) != 0:
            index_coll_4[str(k)] = dict(zip(ll_k, ll_v))
            k = k+1
        if len(vv_v) != 0 and len(vv_v) != 1: 
            all_dict_4 = dict()
            all_dict_4 = dict(zip(vv_k, vv_v))
        else:
            break    
    if len(vv_v) != 0:       
        index_coll_4[str(k)] = dict(zip(vv_k, vv_v))

#%%
    s_name = '5'
    # data_dir = f'data/{dir_s}/{s_d}/{str_f}/{s_n}/{s_name}'
    data_dir = os.path.join(lgdcnn_dir,"data", dir_s, s_d, str_f,s_n,s_name)
    if isinstance(data_dir, list):
        mat_props = data_dir
    elif os.path.isdir(data_dir):
        mat_props = os.listdir(data_dir)
    else:
        mat_props = [data_dir]
    name1 = [i.split('.csv')[0] for i in mat_props if i.split('.csv')[0] == 'fraction_fomula']
    path_formula = os.path.join(data_dir, f'{name1[0]}.csv')
    df_f = pd.read_csv(path_formula)
    formu_5 = df_f['formula'].values
    row_ind_5 = 5
    save_num_5 = np.zeros((len(mat_props)-1, row_ind_5, row_ind_5))
    for i, mat_prop in enumerate(mat_props):
        if mat_prop.split('.csv')[0] == 'fraction_fomula':
            mat_props.pop(i)
    for i, mat_prop in enumerate(mat_props):   
        if os.path.isfile(mat_prop):
            name = os.path.basename(mat_prop).split('.csv')[0]
            path = mat_prop
        elif os.path.isdir(data_dir):
            name = mat_prop.split('.csv')[0]
            path = os.path.join(data_dir, f'{name}.csv')
        else:
            path = data_dir
            name = mat_prop.split('/')[-1].split('.csv')[0]
        # read csv files
        df = pd.read_csv(path)
        if 'element' not in df.columns:
            df['element'] = df.iloc[:,0]
            df = df.drop(columns=['Unnamed: 0'])
        # get rid of 'Null' element representation
        null_row = df['element'] == 'Null'
        df = df[~null_row]
        # grab only the elements that are in oliynyk
        df.index = df['element']
        elements = df.pop('element')
        # normalize the dataframe
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        current_elements = elements
        df_extended = df.reindex(current_elements)
        dfT = df_extended.T
        dfT.columns = current_elements
        corr2 = dfT.corr()
        save_num_5[i,:,:] = corr2.values
    all_dict_5 = dict(zip(formu_5, save_num_5)) 
    index_coll_5 = collections.OrderedDict()
    k = 0
    for k1,values in all_dict_5.items():
        ll_k = []
        ll_v = []
        vv_k = []
        vv_v = []
        for k2,value_q in all_dict_5.items():
            if np.abs(np.subtract(values, value_q)).max() <= thred_v:
                ll_v.append(value_q)
                ll_k.append(k2)
            else:
                vv_v.append(value_q)
                vv_k.append(k2)
        if len(ll_v) != 0:
            index_coll_5[str(k)] = dict(zip(ll_k, ll_v))
            k = k+1
        if len(vv_v) != 0 and len(vv_v) != 1: 
            all_dict_5 = dict()
            all_dict_5 = dict(zip(vv_k, vv_v))
        else:
            break
    if len(vv_v) != 0:       
        index_coll_5[str(k)] = dict(zip(vv_k, vv_v))

#%%
    s_name = '6'
    data_dir = os.path.join(lgdcnn_dir,"data", dir_s, s_d, str_f,s_n,s_name)
    if isinstance(data_dir, list):
        mat_props = data_dir
    elif os.path.isdir(data_dir):
        mat_props = os.listdir(data_dir)
    else:
        mat_props = [data_dir]
    name1 = [i.split('.csv')[0] for i in mat_props if i.split('.csv')[0] == 'fraction_fomula']
    path_formula = os.path.join(data_dir, f'{name1[0]}.csv')
    df_f = pd.read_csv(path_formula)
    formu_6 = df_f['formula'].values
    row_ind_6 = 6
    save_num_6 = np.zeros((len(mat_props)-1, row_ind_6, row_ind_6))
    for i, mat_prop in enumerate(mat_props):
        if mat_prop.split('.csv')[0] == 'fraction_fomula':
            mat_props.pop(i)
    for i, mat_prop in enumerate(mat_props):   
        if os.path.isfile(mat_prop):
            name = os.path.basename(mat_prop).split('.csv')[0]
            path = mat_prop
        elif os.path.isdir(data_dir):
            name = mat_prop.split('.csv')[0]
            path = os.path.join(data_dir, f'{name}.csv')
        else:
            path = data_dir
            name = mat_prop.split('/')[-1].split('.csv')[0]
        # read csv files
        df = pd.read_csv(path)
        if 'element' not in df.columns:
            df['element'] = df.iloc[:,0]
            df = df.drop(columns=['Unnamed: 0'])
        # get rid of 'Null' element representation
        null_row = df['element'] == 'Null'
        df = df[~null_row]
        # grab only the elements that are in oliynyk
        df.index = df['element']
        elements = df.pop('element')
        # normalize the dataframe
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        current_elements = elements
        df_extended = df.reindex(current_elements)
        dfT = df_extended.T
        dfT.columns = current_elements
        corr2 = dfT.corr()
        save_num_6[i,:,:] = corr2.values
    all_dict_6 = dict(zip(formu_6, save_num_6)) 
    index_coll_6 = collections.OrderedDict()
    k = 0
    for k1,values in all_dict_6.items():
        ll_k = []
        ll_v = []
        vv_k = []
        vv_v = []
        for k2,value_q in all_dict_6.items():
            if np.abs(np.subtract(values, value_q)).max() <= thred_v:
                ll_v.append(value_q)
                ll_k.append(k2)
            else:
                vv_v.append(value_q)
                vv_k.append(k2)
        if len(ll_v) != 0:
            index_coll_6[str(k)] = dict(zip(ll_k, ll_v))
            k = k+1
        if len(vv_v) != 0 and len(vv_v) != 1: 
            all_dict_6 = dict()
            all_dict_6 = dict(zip(vv_k, vv_v))
        else:
            break
    if len(vv_v) != 0:       
        index_coll_6[str(k)] = dict(zip(vv_k, vv_v))
   
#%%
    s_name = '7'
    data_dir = os.path.join(lgdcnn_dir,"data", dir_s, s_d, str_f,s_n,s_name)
    if isinstance(data_dir, list):
        mat_props = data_dir
    elif os.path.isdir(data_dir):
        mat_props = os.listdir(data_dir)
    else:
        mat_props = [data_dir]
    name1 = [i.split('.csv')[0] for i in mat_props if i.split('.csv')[0] == 'fraction_fomula']
    path_formula = os.path.join(data_dir, f'{name1[0]}.csv')
    df_f = pd.read_csv(path_formula)
    formu_7 = df_f['formula'].values
    row_ind_7 = 7
    save_num_7 = np.zeros((len(mat_props)-1, row_ind_7, row_ind_7))

    for i, mat_prop in enumerate(mat_props):
        if mat_prop.split('.csv')[0] == 'fraction_fomula':
            mat_props.pop(i)
    for i, mat_prop in enumerate(mat_props):   
        if os.path.isfile(mat_prop):
            name = os.path.basename(mat_prop).split('.csv')[0]
            path = mat_prop
        elif os.path.isdir(data_dir):
            name = mat_prop.split('.csv')[0]
            path = os.path.join(data_dir, f'{name}.csv')
        else:
            path = data_dir
            name = mat_prop.split('/')[-1].split('.csv')[0]
        # read csv files
        df = pd.read_csv(path)
        if 'element' not in df.columns:
            df['element'] = df.iloc[:,0]
            df = df.drop(columns=['Unnamed: 0'])
        # get rid of 'Null' element representation
        null_row = df['element'] == 'Null'
        df = df[~null_row]
        # grab only the elements that are in oliynyk
        df.index = df['element']
        elements = df.pop('element')
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        current_elements = elements
        df_extended = df.reindex(current_elements)
        dfT = df_extended.T
        dfT.columns = current_elements
        corr2 = dfT.corr()
        save_num_7[i,:,:] = corr2.values
    all_dict_7 = dict(zip(formu_7, save_num_7)) 
    index_coll_7 = collections.OrderedDict()
    k = 0
    for k1,values in all_dict_7.items():
        ll_k = []
        ll_v = []
        vv_k = []
        vv_v = []
        for k2,value_q in all_dict_7.items():
            if np.abs(np.subtract(values, value_q)).max() <= thred_v:
                ll_v.append(value_q)
                ll_k.append(k2)
            else:
                vv_v.append(value_q)
                vv_k.append(k2)
        if len(ll_v) != 0:
            index_coll_7[str(k)] = dict(zip(ll_k, ll_v))
            k = k+1
        if len(vv_v) != 0 and len(vv_v) != 1: 
            all_dict_7 = dict()
            all_dict_7 = dict(zip(vv_k, vv_v))
        else:
            break 
    if len(vv_v) != 0:       
        index_coll_7[str(k)] = dict(zip(vv_k, vv_v))
  
    corr_conc = len(index_coll) + len(index_coll_4)+len(index_coll_5)+len(index_coll_6)+len(index_coll_7)
    print(f'{num}-----{thred_v}:  {corr_conc}')
