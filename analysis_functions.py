import os
import jupyter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm.auto import tqdm
import scipy
from scipy.stats import sem
import matplotlib
from scipy.stats import ttest_1samp
from scipy.stats import pearsonr
matplotlib.rcParams['pdf.fonttype'] = 42 


def get_data(data_path ='' ,recalculate = False):
    RT_outlier_max = 7
    RT_outlier_min = 1.5
    
    to_pandas = defaultdict(list)

    os.makedirs("outputs",exist_ok=True)
    if recalculate is False and os.path.exists("outputs/cache_data.pkl"):
        print("Loading from cache")
        df = pd.read_pickle("outputs/cache_data.pkl")
        return df

    all_paths = [f"{data_path}1. COMPETITION_data_raw_static/", f"{data_path}2. COMPETITION_data_raw_dynamic/"]
    
    print("Loading Data")
    for b_path_d in all_paths:
        sceds = np.sort(os.listdir(b_path_d))
        sceds  = [sced for sced in sceds if 'schedule_' in sced or 'data_raw_dynamic_data_files' in sced]
        for sced in (pbar:= tqdm(sceds)):
            pbar.set_description(f"Loading {sced}")
            for fl in os.listdir(b_path_d +'/' + sced):
                if '.csv' not in fl:                          
                    continue

                data = pd.read_csv(b_path_d +'/' + sced +'/' +fl)

                # Fix  3 problematic files, without data in columns
                if fl == '1609704495_75.csv':
                    data['time'] = '01-03-2021 00:' + data['time']
                if fl == '1609704285_30.csv':
                    data['time'] = '01-03-2021 00:' +data['time'] 
                if fl == '1610553433_92.csv':
                    data['time'] = '01-13-2021 00:'+ data['time']
                

                to_pandas['participant_n'].append(fl.split('_')[-1][:-4])
                if 'data_raw_dynamic' in b_path_d:
                    to_pandas['sced'].append(data['schedule_name'].iloc[0])
                else:
                    to_pandas['sced'].append(sced)
                to_pandas['path_name'].append(sced +'/' +fl)
                

                to_pandas['biased_m'].append(sum(data['is_biased_choice'])/100)
                to_pandas['detected_rewards_B'].append(len(data.query('is_biased_choice==True & biased_reward==1')))
                to_pandas['detected_rewards_UB'].append(len(data.query('is_biased_choice==False & unbiased_reward==1')))
                

                data['is_choice_1'] = 0
                data['won_reward_at_1'] = 0   
                data['won_reward_at_2'] = 0   
                data['total_reward_allocated'] = 0 
                data['won_any_reward'] = 0   
                data.loc[data.query('is_biased_choice==1').index,'is_choice_1'] = 1
                data.loc[data.query('is_biased_choice==True & biased_reward==1').index,'won_reward_at_1'] = 1 
                data.loc[data.query('is_biased_choice==False & unbiased_reward==1').index,'won_reward_at_2'] = 1 
                data['won_any_reward'] = data['won_reward_at_1'] + data['won_reward_at_2']
                data['total_reward_allocated'] = data['observed_reward']
                
                
                # Response time calcualtions
                date_day = pd.to_datetime(data['time'].iloc[0].split()[0])
                date_full = pd.to_datetime(data['time'].iloc[0])
                date_raw = data['time'].iloc[0]
                
                to_pandas['date_day'].append(date_day)
                to_pandas['date_full'].append(date_full)
                to_pandas['date_raw'].append(date_raw)
            
                ## response time
                data['rt'] = pd.to_datetime(data['time']).diff().apply(lambda x:x.total_seconds())
                data['rt_raw'] = data['rt'].copy()
                data.loc[data.query('rt>@RT_outlier_max').index.values,'rt'] = None # remove response time larger than 7 seconds
                data.loc[data.query('rt<@RT_outlier_min').index.values,'rt'] = None # remove response time smaller than 1.5 s
                
                rt_mean = data.groupby("is_biased_choice")['rt'].mean()
                
                
                if True in rt_mean and False in rt_mean: # Cases with no response time for each direction where removes, as we could not calculate their difference
                    to_pandas['rt_B'].append(rt_mean[True])
                    to_pandas['rt_UB'].append(rt_mean[False])

                else:
                    to_pandas['rt_B'].append(None)
                    to_pandas['rt_UB'].append(None)
                
                to_pandas['N_rt_outliers'].append(len(data.query('rt_raw>@RT_outlier_max').index.values) + len(data.query('rt_raw<@RT_outlier_min').index.values))



                to_pandas['data'].append(data)
    

    df = pd.DataFrame(to_pandas)
    df['rewards_total'] = df['detected_rewards_B']+df['detected_rewards_UB']
    df['B-UB'] = df['detected_rewards_B']-df['detected_rewards_UB']
    df['B/(UB+1)'] = df['detected_rewards_B']/(df['detected_rewards_UB']+1) # added 1 to avoid inf/NANS.
    df['B/(B+UB)'] = df['detected_rewards_B']/(df['detected_rewards_B']+df['detected_rewards_UB'])
    df['UB/(B+UB)'] = df['detected_rewards_UB']/(df['detected_rewards_B']+df['detected_rewards_UB'])
    df['(B-UB)/(B+UB)'] = (df['detected_rewards_B']-df['detected_rewards_UB'])/(df['detected_rewards_B']+df['detected_rewards_UB'])
    df['rewards_freq_b'] = df['detected_rewards_B']/(df['biased_m']*100)
    df['rewards_freq_ub'] = df['detected_rewards_UB']/((1-df['biased_m'])*100) 

    df['rt_B_m_UB'] = df['rt_B'] - df['rt_UB']

    df.to_pickle("outputs/cache_data.pkl")
    return df



def post_process(df):
    def change_name(x):
        if len(x)<12:
            return f'SS{int(x.split("_")[1]):02}'
        else:
            return f'DS{int(x.split("_")[1]):02}'
    
    # change names of schedules to SS or DS:
    df['sced'] = df['sced'].apply(change_name)

    # combine the two dynamic schedules
    df_s1 = df.query("sced=='DS01'").copy()
    df_s2 = df.query("sced=='DS02'").copy()
    df_s1['sced'] = 'RaCaS'
    df_s2['sced'] = 'RaCaS'
    df = pd.concat([df_s1,df_s2,df])

    # add a new sced which is the pooled version of all the others without SS00 and DS01, DS02 and RaCaS
    all_temp = []
    for sced in df.sced.unique():
        if sced not in ['RaCaS','SS00', 'DS01', 'DS02']:
            tm = df.query("sced==@sced").copy()
            tm['sced'] = 'Control (pooled)'
            all_temp.append(tm)
    
    df = pd.concat(all_temp + [df])


    # new column names
    column_names = {'sced':'Model/Schedule', 'biased_m':'Bias', 'detected_rewards_B':'Detected Rewards (B+)', 'detected_rewards_UB':'Detected Rewards (B-)'
                    ,'B-UB':'$Δ_{Rewards}$:[(B+)-(B-)]', '(B-UB)/(B+UB)':'$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]', 'rewards_freq_b':'Observed Expectancy (B+)', 
                    'rewards_freq_ub':'Observed Expectancy (B-)', 'rt_B_m_UB': '$Δ_{RT}$:[(B+)-(B-)]', 'participant_n':'N'}
    return df.rename(columns=column_names, inplace=False)


def create_summary_table(df):

    H0_per_column = {'Bias':0.5, 'Detected Rewards (B+)':12.5, 'Detected Rewards (B-)':12.5, '$Δ_{Rewards}$:[(B+)-(B-)]':0, 
                     '$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]':0,'Observed Expectancy (B+)':0.25, 'Observed Expectancy (B-)':0.25,
                    '$Δ_{RT}$:[(B+)-(B-)]':0}
    
    df_summary = df.groupby("Model/Schedule").mean().sort_values("Bias",ascending=False) # calculate mean per schedule and sort by bias
    df_count = df.groupby("Model/Schedule").count()
    df_summary['N'] = df_count['N'] # add number of participants per schedule

    # change sort
    manual_sort_start = ['RaCaS','Control (pooled)']
    manual_sort_end = ['DS01','DS02', 'SS00']
    sceds_names_sorted = manual_sort_start  + [i for i in df_summary.index if i not in manual_sort_start+manual_sort_end] + manual_sort_end
    df_summary = df_summary.loc[sceds_names_sorted]


    df_std = df.groupby("Model/Schedule").std()
    df_pval = df.groupby("Model/Schedule").apply(lambda x: pd.Series({i:ttest_1samp(x[i],H0_per_column[i],nan_policy='omit').pvalue for i in H0_per_column}))



    out = pd.concat([df_summary, df_std,df_pval]).groupby(level=0).agg(clean_cell_3)

    out['N'] = df_count['N']

    # out.index.names = [column_names['Model/Schedule']]
    #out = out.rename(columns=column_names, index=column_names).loc[:,list(column_names.values())[1:]]

    #Bias vs. CI (Cor.), Bias vs. Total Detected Rewards (Cor.)

    #Bias vs. Delta RT (Cor.)
    out_full = out.loc[sceds_names_sorted]
    out_condensed = out_full.loc[:,list(H0_per_column.keys())]

    r = df.groupby('Model/Schedule')[['Bias','$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]']].corr(method=lambda x, y: pearsonr(x, y)[0]).unstack().iloc[:,1]
    p = df.groupby('Model/Schedule')[['Bias','$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]']].corr(method=lambda x, y: pearsonr(x, y)[1]).unstack().iloc[:,1]
    out['Bias vs. Norm. $Δ_{Rewards}$ (Cor.)'] = pd.concat([r, p]).groupby(level=0).agg(clean_cell_2)
    out_condensed['Bias vs. Norm. $Δ_{Rewards}$ (Cor.)'] = out['Bias vs. Norm. $Δ_{Rewards}$ (Cor.)']
    
    r = df.groupby('Model/Schedule')[['Bias','rewards_total']].corr(method=lambda x, y: pearsonr(x, y)[0]).unstack().iloc[:,1]
    p = df.groupby('Model/Schedule')[['Bias','rewards_total']].corr(method=lambda x, y: pearsonr(x, y)[1]).unstack().iloc[:,1]
    out['Bias vs. Total Rewards (Cor.)'] = pd.concat([r, p]).groupby(level=0).agg(clean_cell_2)
    out_condensed['Bias vs. Total Rewards (Cor.)'] = out['Bias vs. Total Rewards (Cor.)']

    r = df.groupby('Model/Schedule')[['Bias','$Δ_{RT}$:[(B+)-(B-)]']].corr(method=lambda x, y: pearsonr(x, y)[0]).unstack().iloc[:,1]
    p = df.groupby('Model/Schedule')[['Bias','$Δ_{RT}$:[(B+)-(B-)]']].corr(method=lambda x, y: pearsonr(x, y)[1]).unstack().iloc[:,1]
    out['Bias vs. Delta RT (Cor.)'] = pd.concat([r, p]).groupby(level=0).agg(clean_cell_2)
    out_condensed['Bias vs. Delta RT (Cor.)'] = out['Bias vs. Delta RT (Cor.)']

    return out_full, out_condensed
    #out.to_excel('new_data_v4.xlsx')


def collect_rolling_avereage(df, rw=10):

    rolling_data = {}
    # Collect data
    for sced in (pbar:= tqdm(df['Model/Schedule'].unique())):
        pbar.set_description(f"Processing {sced}")
        data_per_sced = defaultdict(list)

        for _, row in df.query('`Model/Schedule`==@sced').iterrows():
            data_per_sced['choice_biased'].append(row['data']['is_choice_1'].rolling(rw, min_periods=0).mean().values)

            data_per_sced['reward_in_1'].append(row['data']['biased_reward'].rolling(rw, min_periods=0).mean().values)
            data_per_sced['reward_in_2'].append(row['data']['unbiased_reward'].rolling(rw, min_periods=0).mean().values)
            data_per_sced['allocated_rewards_total'].append(row['data']['total_reward_allocated'].rolling(rw, min_periods=0).mean().values)

            data_per_sced['recived_reward_biased'].append(row['data']['won_reward_at_1'].rolling(rw, min_periods=0).mean().values)
            data_per_sced['recived_reward_non_biased'].append(row['data']['won_reward_at_2'].rolling(rw, min_periods=0).mean().values)
            data_per_sced['recived_reward_total'].append(row['data']['won_any_reward'].rolling(rw, min_periods=0).mean().values)

            temp_1 = row['data']['biased_reward'].rolling(rw,min_periods=0).mean().values
            temp_1b = row['data']['won_reward_at_1'].rolling(rw,min_periods=0).mean().values
            temp_2 = row['data']['unbiased_reward'].rolling(rw,min_periods=0).mean().values
            temp_2b = row['data']['won_reward_at_2'].rolling(rw,min_periods=0).mean().values
            temp_3 = row['data']['total_reward_allocated'].rolling(rw,min_periods=0).mean().values
            temp_3b = row['data']['won_any_reward'].rolling(rw,min_periods=0).mean().values

            aloc_to_disc_ratio_1=  temp_1b/temp_1
            aloc_to_disc_ratio_2=  temp_2b/temp_2
            aloc_to_disc_ratio_total = temp_3b/temp_3


            data_per_sced['recived_reward_biased_ratio'].append(aloc_to_disc_ratio_1)
            data_per_sced['recived_reward_non_biased_ratio'].append(aloc_to_disc_ratio_2)
            data_per_sced['recived_reward_ratio'].append(aloc_to_disc_ratio_total)
            
            
            won_reward_b = row['data']['won_reward_at_1'].cumsum()
            choice_reward_b = row['data']['is_choice_1'].cumsum()
            data_per_sced['actual_evidence_b'].append(won_reward_b/choice_reward_b)

            won_reward_ub = row['data']['won_reward_at_2'].cumsum()
            choice_reward_ub = (~(row['data']['is_choice_1']==1)).astype(int).cumsum()
            data_per_sced['actual_evidence_ub'].append(won_reward_ub/choice_reward_ub)


            won_reward_total = row['data']['won_any_reward'].cumsum()
            data_per_sced['actual_evidence_total'].append(won_reward_total/(1+np.arange(len(won_reward_b))))


            temp_b = [25] + list(25 - row['data']['biased_reward'].cumsum())
            data_per_sced['actual_expectency_b'].append(temp_b[:-1]/(100-(np.arange(0,100)))) 
            temp_b = [25] + list(25 - row['data']['unbiased_reward'].cumsum())
            data_per_sced['actual_expectency_ub'].append(temp_b[:-1]/(100-(np.arange(0,100))))
            
            temp_b = [25] + list(25 - row['data']['won_reward_at_1'].cumsum())
            data_per_sced['observed_expectency_b'].append(temp_b[:-1]/(100-(np.arange(0,100)))) 
            temp_b = [25] + list(25 - row['data']['won_reward_at_2'].cumsum())
            data_per_sced['observed_expectency_ub'].append(temp_b[:-1]/(100-(np.arange(0,100))))
            
            dynamic_a_minus_b_over_a_plus_b = (row['data']['won_reward_at_1'].cumsum() - row['data']['won_reward_at_2'].cumsum())/(row['data']['won_reward_at_1'].cumsum() + row['data']['won_reward_at_2'].cumsum())
            data_per_sced['dynamic_a_minus_b_over_a_plus_b'].append(dynamic_a_minus_b_over_a_plus_b)
            
            # rolling version
            dynamic_a_minus_b_over_a_plus_b_r = (row['data']['won_reward_at_1'].rolling(rw,min_periods=0).sum() - row['data']['won_reward_at_2'].rolling(rw,min_periods=0).sum())/(row['data']['won_reward_at_1'].rolling(rw,min_periods=0).sum() + row['data']['won_reward_at_2'].rolling(rw,min_periods=0).sum())
            data_per_sced['dynamic_a_minus_b_over_a_plus_b_r'].append(dynamic_a_minus_b_over_a_plus_b_r)

            
            
        rolling_data[sced] = data_per_sced
    return rolling_data





def Figure_2(df_noPool_noSS00_noRaCaS, df_Pool_RaCaS, rolling_data):

    df_noPool_noSS00_noRaCaS = df_noPool_noSS00_noRaCaS.copy()
    df_Pool_RaCaS = df_Pool_RaCaS.copy()

    sced_sort = df_noPool_noSS00_noRaCaS.groupby('Model/Schedule')['Bias'].mean().sort_values(ascending=False).index
    df_noPool_noSS00_noRaCaS['Model/Schedule'] = df_noPool_noSS00_noRaCaS['Model/Schedule'].astype("category")
    df_noPool_noSS00_noRaCaS['Model/Schedule'] = df_noPool_noSS00_noRaCaS['Model/Schedule'].cat.set_categories(sced_sort)
    colors = ['tab:olive','tab:gray']


    plt.figure(figsize=(25,5))
    # A
    sns.boxplot(x='Model/Schedule',y='Bias',data=df_noPool_noSS00_noRaCaS,ax=plt.gca(),
                palette=[colors[0]]*2 + [colors[1]]*(len(sced_sort)-2), showfliers=False)
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='Bias', data=df_noPool_noSS00_noRaCaS,ax=plt.gca(),color='k',alpha=0.2)
    plt.ylabel("Bias$^{+}$")
    os.makedirs('outputs/figure_2',exist_ok=True)
    plt.savefig(f'outputs/figure_2/A_Sched_vs_Bias.pdf')

    # B
    plt.figure(figsize=(3,5))
    df_RaCaS = df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')
    df_Pooled = df_Pool_RaCaS.query('`Model/Schedule`=="Control (pooled)"')

    bins = np.array(list(range(0,21)))/20
    bins[-1]+=0.000001 # make sure to add the last bin to include the last value

    plt.hist(df_RaCaS['Bias'],bins=bins,weights=np.ones(len(df_RaCaS))/len(df_RaCaS),color=colors[0],
             alpha=0.8,edgecolor='k',linewidth=1,label='RaCaS')
    plt.hist(df_Pooled['Bias'],bins=bins,weights=np.ones(len(df_Pooled))/len(df_Pooled),color=colors[1],
             alpha=0.5,edgecolor='k',linewidth=1,label="Control (pooled)")
    plt.axvline(0.5,ls='--',color='brown')
    plt.xlabel("Bias$^{+}$")
    plt.ylabel("Proportion of participants")
    plt.legend()
    plt.savefig(f'outputs/figure_2/B_Bias_histogram.pdf')

    # C

    cmap = plt.cm.get_cmap('gray')
    cmap.set_under('white')
    cmap.set_over('black')
    plt.figure(figsize=(9,5))
    matrix_a = []
    for _, row in df_RaCaS.sort_values('Bias',ascending=True).iterrows():
        matrix_a.append(list(row['data']['is_choice_1'].values))
    
    plt.pcolormesh(matrix_a,cmap=cmap, vmin=0.5, vmax = 0.7)
    plt.xlabel("Trial")
    plt.ylabel("Subject (sorted by bias$^{+}$)")
    plt.title(u'Choice is 1 (■) or 2 (□)')
    plt.tight_layout()
    plt.savefig(f'outputs/figure_2/C_Choice_per_subject.pdf')

    # D
    plt.figure(figsize=(12,4))
    plt.ylim([0,1])
    sns.set_style(style="white")
    plot_with_sem_winning('choice_biased', rolling_data['RaCaS'],lw=3,label='RaCaS',ls='-',color='black')
    plot_with_sem_winning('choice_biased', rolling_data['Control (pooled)'],lw=3,label='Control (pooled)',ls='--',color='black')
    plt.axhline(0.5,ls='--',color='brown')
    plt.legend()
    plt.xlabel('Trial')
    plt.ylabel('Bias$^{+}$ (Rolling Average)')
    sns.despine()
    plt.savefig(f'outputs/figure_2/D_Choice_per_subject.pdf')

    print("DS01, mean = " + str(df_noPool_noSS00_noRaCaS.query('`Model/Schedule`=="DS01"')['Bias'].mean()), 'std = ' + str(df_noPool_noSS00_noRaCaS.query('`Model/Schedule`=="DS01"')['Bias'].std()))
    print("DS02, mean = " + str(df_noPool_noSS00_noRaCaS.query('`Model/Schedule`=="DS02"')['Bias'].mean()), 'std = ' + str(df_noPool_noSS00_noRaCaS.query('`Model/Schedule`=="DS02"')['Bias'].std()))
    print("RaCaS, mean = " + str(df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['Bias'].mean()), 'std = ' + str(df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['Bias'].std()))
    

def Figure_3(df_noPool_noSS00_noDS01_noDS02, df_Pool_RaCaS, rolling_data):

    df_noPool_noSS00_noDS01_noDS02 = df_noPool_noSS00_noDS01_noDS02.copy()
    df_Pool_RaCaS = df_Pool_RaCaS.copy()

    sced_sort = df_noPool_noSS00_noDS01_noDS02.groupby('Model/Schedule')['Bias'].mean().sort_values(ascending=False).index
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].astype("category")
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].cat.set_categories(sced_sort)
    colors = ['tab:olive','tab:gray']
    
    # A
    plt.figure(figsize=(25,5))
    sns.boxplot(x='Model/Schedule',y='$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]',data=df_noPool_noSS00_noDS01_noDS02,ax=plt.gca(),
                palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1), showfliers=False)
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]', data=df_noPool_noSS00_noDS01_noDS02,ax=plt.gca(),color='k',alpha=0.2)
    os.makedirs('outputs/figure_3',exist_ok=True)
    plt.savefig(f'outputs/figure_3/A_Sched_vs_delta_R_norm.pdf')

    # B
    plt.figure(figsize=(3,5))
    df_RaCaS = df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')
    df_Pooled = df_Pool_RaCaS.query('`Model/Schedule`=="Control (pooled)"')
    bins = np.array(list(range(-10,11)))/10
    bins[-1]+=0.000001 # make sure to add the last bin to include the last value

    plt.hist(df_RaCaS['$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]'],bins=bins,weights=np.ones(len(df_RaCaS))/len(df_RaCaS),color=colors[0],
             alpha=0.8,edgecolor='k',linewidth=1,label='RaCaS')
    plt.hist(df_Pooled['$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]'],bins=bins,weights=np.ones(len(df_Pooled))/len(df_Pooled),color=colors[1],
             alpha=0.5,edgecolor='k',linewidth=1,label="Control (pooled)")
    plt.axvline(0, ls='--',color='brown')
    plt.xlabel('$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]')
    plt.ylabel("Proportion of participants")
    plt.legend()
    plt.savefig(f'outputs/figure_3/B_delta_R_norm_histogram.pdf')


    # C
    plt.figure(figsize=(7,7),dpi=150)
    cmap_r = plt.cm.get_cmap('Reds'); cmap_r.set_under('white'); cmap_r.set_over('red')

    cmap_b = plt.cm.get_cmap('Blues'); cmap_b.set_under('white'); cmap_b.set_over('blue')

    matrix_at_1 = []
    matrix_at_2 = []

    for _, row in df_RaCaS.sort_values('Bias',ascending=False).iterrows():
        matrix_at_1.append(list(row['data']['won_reward_at_1'].values))
        matrix_at_2.append(list(row['data']['won_reward_at_2'].values))

    matrix_at_2 = np.array(matrix_at_2).astype(float)
    matrix_at_2[matrix_at_2==0] = np.nan
    plt.imshow(matrix_at_1,cmap=cmap_r, vmin=0.5, vmax = 0.7,aspect='auto')
    plt.imshow(matrix_at_2,cmap=cmap_b, vmin=0.5, vmax = 0.7,aspect='auto')
    plt.xlabel("Trial")
    plt.ylabel("Subject (sorted by bias)")
    plt.title(u'Choice is 1 (red) or 2 (blue)')
    plt.tight_layout()
    plt.savefig(f'outputs/figure_3/C_Choice_per_subject.pdf')
    # D + E
    plt.figure(figsize=(12,4))

    sns.set_style(style="white")
    delta_R_norm = 'dynamic_a_minus_b_over_a_plus_b_r'
    plot_with_sem_winning('choice_biased', rolling_data['RaCaS'],lw=3,label='RaCaS - Bias$^{+}$',ls='-',color='gray')
    plot_with_sem_winning(delta_R_norm, rolling_data['RaCaS'],lw=3,label='RaCaS - $Δ_{Rewards}$ (Norm.)',ls='-',color='purple')

    plot_with_sem_winning('choice_biased', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Bias$^{+}$',ls='--',color='gray')
    plot_with_sem_winning(delta_R_norm, rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - $Δ_{Rewards}$ (Norm.)',ls='--',color='purple')
    plt.ylim([-1,1.02])
    plt.xlim(0,100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Trial')
    plt.ylabel('Mean Rolling Average')
    sns.despine()
    plt.savefig(f'outputs/figure_3/D_dynamic_reward_norm.pdf')

    plt.figure(figsize=(12,4))
    plot_with_sem_winning('choice_biased', rolling_data['RaCaS'],lw=3,label='RaCaS - Bias$^{+}$',ls='-',color='gray')
    plot_with_sem_winning('actual_evidence_b', rolling_data['RaCaS'],lw=3,label='RaCaS - Expectancy (B$^{+}$)',ls='-',color='red')
    plot_with_sem_winning('actual_evidence_ub', rolling_data['RaCaS'],lw=3,label='RaCaS - Expectancy (B$^{-}$)',ls='-',color='blue')
    plot_with_sem_winning('actual_evidence_total', rolling_data['RaCaS'],lw=3,label='RaCaS - Expectancy (Tot.)',ls='-',color='purple')


    plot_with_sem_winning('choice_biased', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Bias$^{+}$',ls='--',color='gray')
    plot_with_sem_winning('actual_evidence_b', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Expectancy (B$^{+}$)',ls='--',color='red')
    plot_with_sem_winning('actual_evidence_ub', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Expectancy (B$^{-}$)',ls='--',color='blue')
    plot_with_sem_winning('actual_evidence_total', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Expectancy (Tot.)',ls='--',color='purple')

    plt.ylim([0,1])
    plt.xlim(0,100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Trial')
    plt.ylabel('Mean')
    sns.despine()
    plt.savefig(f'outputs/figure_3/D_dynamic_expectancy.pdf')

    # print delta rward norm for RaCaS
    print("RaCaS, mean = " + str(df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]'].mean()), 'std = ' + str(df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]'].std()))
    # now print median
    print("RaCaS, median = " + str(df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['$Δ_{Rewards}$ (Norm.):[(B+)-(B-)]:[(B+)-(B-)]'].median()))

def Figure_4(df_noPool_noSS00_noDS01_noDS02, df_Pool_RaCaS, rolling_data):

    df_noPool_noSS00_noDS01_noDS02 = df_noPool_noSS00_noDS01_noDS02.copy()
    df_Pool_RaCaS = df_Pool_RaCaS.copy()
    os.makedirs('outputs/figure_4',exist_ok=True)

    sced_sort = df_noPool_noSS00_noDS01_noDS02.groupby('Model/Schedule')['Bias'].mean().sort_values(ascending=False).index
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].astype("category")
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].cat.set_categories(sced_sort)
    colors = ['tab:olive','tab:gray']
    
    # A 
    df_RaCaS = df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')
    df_Pooled = df_Pool_RaCaS.query('`Model/Schedule`=="Control (pooled)"')

    plt.figure(figsize=(4,4))
    plt.scatter(x=df_Pooled['Bias'],y=df_Pooled['Detected Rewards (B+)'] + df_Pooled['Detected Rewards (B-)'],color=colors[1],alpha=0.8,
                label='Control (pooled)',linewidth=0.2,edgecolor='k')
    plt.scatter(x=df_RaCaS['Bias'],y=df_RaCaS['Detected Rewards (B+)'] + df_RaCaS['Detected Rewards (B-)'],color=colors[0],alpha=0.8,
                label='RaCaS',linewidth=0.2,edgecolor='k')
    plt.axhline(25,ls='--',color='brown')
    plt.axvspan(0.4,0.6,alpha=0.2,color='red')
    plt.xlabel('Bias')
    plt.ylabel('Total Rewards Won')
    plt.savefig(f'outputs/figure_4/A_Bias_vs_total_rewards.pdf')

    # B
    plt.figure(figsize=(4,4))
    plt.hist(df_Pooled['Detected Rewards (B+)'] + df_Pooled['Detected Rewards (B-)'],bins=range(10,51,2),weights=np.ones(len(df_Pooled))/len(df_Pooled),color=colors[1],
                alpha=0.8,edgecolor='k',linewidth=1,label='Control (pooled)')
    plt.hist(df_RaCaS['Detected Rewards (B+)'] + df_RaCaS['Detected Rewards (B-)'],bins=range(10,51,2),weights=np.ones(len(df_RaCaS))/len(df_RaCaS),color=colors[0],
                alpha=0.6,edgecolor='k',linewidth=1,label='RaCaS')
    plt.axvline(25,ls='--',color='brown')
    plt.xlabel('Total Rewards Won')
    plt.ylabel('Proportion of participants')
    plt.savefig(f'outputs/figure_4/B_hist_total_rewards.pdf')

    # C
    plt.figure(figsize=(25,5))
    df_noPool_noSS00_noDS01_noDS02['Total_rewards'] = df_noPool_noSS00_noDS01_noDS02['Detected Rewards (B+)'] + df_noPool_noSS00_noDS01_noDS02['Detected Rewards (B-)']
    sns.boxplot(x='Model/Schedule',y='Total_rewards',data=df_noPool_noSS00_noDS01_noDS02,ax=plt.gca(),
                palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1), showfliers=False)
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='Total_rewards', data=df_noPool_noSS00_noDS01_noDS02,ax=plt.gca(),color='k',alpha=0.2)
    plt.axhline(25,ls='--',color='brown')
    plt.ylabel('Total Rewards Won')
    plt.savefig(f'outputs/figure_4/C_Scheds_total_Rewards.pdf')


    # D
    plt.figure(figsize=(12,4))
    plt.ylim([0,1])
    sns.set_style(style="white")
    plot_with_sem_winning('choice_biased', rolling_data['RaCaS'],lw=3,label='RaCaS - Bias',ls='-',color='gray')
    plot_with_sem_winning('recived_reward_biased_ratio', rolling_data['RaCaS'],lw=3,label='RaCaS - Exploitation(B+)',ls='-',color='red')
    plot_with_sem_winning('recived_reward_non_biased_ratio', rolling_data['RaCaS'],lw=3,label='RaCaS - Exploitation(B-)',ls='-',color='blue')

    plot_with_sem_winning('choice_biased', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Bias',ls='--',color='gray')
    plot_with_sem_winning('recived_reward_biased_ratio', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Exploitation(B+)',ls='--',color='red')
    plot_with_sem_winning('recived_reward_non_biased_ratio', rolling_data['Control (pooled)'],lw=3,label='Control (pooled) - Exploitation(B-)',ls='--',color='blue')

    plt.axhline(0.5,ls='--',color='brown')
    plt.xlim(0,100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Trial')
    plt.ylabel('Rolling Average')
    sns.despine()
    plt.savefig(f'outputs/figure_4/D_Dynamic_Exploitation.pdf')

    # print total rewards for RaCaS
    print("RaCaS, mean = " + str((df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['Detected Rewards (B+)'] + df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')['Detected Rewards (B-)']).mean()))

def Sup_Figure_1(df_DS01_DS02):
    df_DS01 = df_DS01_DS02.query('`Model/Schedule`=="DS01"')
    df_DS02 = df_DS01_DS02.query('`Model/Schedule`=="DS02"')
    os.makedirs('outputs/Sup_figure_1',exist_ok=True)
    cmap_k = plt.cm.get_cmap('gray'); cmap_k.set_under('white'); cmap_k.set_over('black')
    cmap_r = plt.cm.get_cmap('Reds'); cmap_r.set_under('white'); cmap_r.set_over('red')
    cmap_b = plt.cm.get_cmap('Blues'); cmap_b.set_under('white'); cmap_b.set_over('blue')

    cmaps = [cmap_k,cmap_r,cmap_b]

    # plot heatmap of choice direction reward won as bias+ and reward won at bias- 
    fig, axes = plt.subplots(3,2,sharex=True,sharey=True,figsize=(22,21),dpi=50)
    title_names = [u'Choice is 1 (■) or 2 (□)', 'Reward won at Bias$^{+}$', 'Reward won at Bias$^{-}$']
    for i, df in enumerate([df_DS01, df_DS02]):
        for j, column in enumerate(['is_choice_1', 'won_reward_at_1', 'won_reward_at_2']):
            matrix = []
            matrix_allocated = []
            for _, row in df.sort_values('Bias',ascending=True).iterrows():
                matrix.append(list(row['data'][column].values))
                if column != 'is_choice_1':
                    if column == 'won_reward_at_1':
                        matrix_allocated.append(list(row['data']['biased_reward'].values))
                    else:
                        matrix_allocated.append(list(row['data']['unbiased_reward'].values))
            matrix = np.array(matrix).astype(float)
            matrix[matrix==0] = np.nan
            axes[j][i].pcolormesh(matrix,cmap=cmaps[j], vmin=0.5, vmax = 0.7)#,aspect='auto')
            axes[j][i].set_title(title_names[j])
            if column != 'is_choice_1':
                axes[j][i].pcolormesh(matrix_allocated,cmap='gray', vmin=0.5, vmax = 0.7,alpha=0.7)

    plt.suptitle('Dynamic Schedule 1                                   Dynamic Schedule 2')
    plt.tight_layout()
    plt.savefig(f'outputs/Sup_figure_1/A_Choice_and_reward_per_subject.pdf')

    # show reward won at bias+ or bias- for each schedule
    plt.figure(figsize=(4,12))
    plt.subplot(2,1,1)
    sns.boxplot(x='Model/Schedule',y='Detected Rewards (B+)',data=df_DS01_DS02)
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='Detected Rewards (B+)', data=df_DS01_DS02,color='k',alpha=0.2)
    plt.axhline(12.5,ls='--',color='brown')

    plt.subplot(2,1,2)
    sns.boxplot(x='Model/Schedule',y='Detected Rewards (B-)',data=df_DS01_DS02)
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='Detected Rewards (B-)', data=df_DS01_DS02,color='k',alpha=0.2)
    plt.axhline(12.5,ls='--',color='brown')
    plt.savefig(f'outputs/Sup_figure_1/B_Rewards_per_schedule.pdf')
    
def Sup_Figure_2(df_noPool_noSS00_noDS01_noDS02, df_Pool_RaCaS):
    os.makedirs('outputs/Sup_figure_2',exist_ok=True)
    df_RaCaS = df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')
    df_Pool = df_Pool_RaCaS.query('`Model/Schedule`=="Control (pooled)"')
    df_noPool_noSS00_noDS01_noDS02 = df_noPool_noSS00_noDS01_noDS02.copy()
    sced_sort = df_noPool_noSS00_noDS01_noDS02.groupby('Model/Schedule')['Bias'].mean().sort_values(ascending=False).index
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].astype("category")
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].cat.set_categories(sced_sort)
    colors = ['tab:olive','tab:gray']

    cmap_r = plt.cm.get_cmap('Reds'); cmap_r.set_under('white'); cmap_r.set_over('red')
    cmap_b = plt.cm.get_cmap('Blues'); cmap_b.set_under('white'); cmap_b.set_over('blue')
    cmap_k = plt.cm.get_cmap('gray'); cmap_k.set_under('white'); cmap_k.set_over('black')

    matrix_won_at_1 = []
    matrix_won_at_2 = []
    matrix_reward_at_1 = []
    matrix_reward_at_2 = []

    for _, row in df_RaCaS.sort_values('Bias',ascending=True).iterrows():
        matrix_won_at_1.append(list(row['data']['won_reward_at_1'].values))
        matrix_won_at_2.append(list(row['data']['won_reward_at_2'].values))
        matrix_reward_at_1.append(list(row['data']['biased_reward'].values))
        matrix_reward_at_2.append(list(row['data']['unbiased_reward'].values))
        
    
    # A
    plt.figure(figsize=(14,14),dpi=150)
    matrix_won_at_1 = np.array(matrix_won_at_1).astype(float)
    matrix_won_at_1[matrix_won_at_1==0] = np.nan

    plt.pcolormesh(matrix_reward_at_1,cmap=cmap_k, vmin=0.5, vmax = 0.7,alpha=0.7)

    plt.pcolormesh(matrix_won_at_1,cmap=cmap_r, vmin=0.5, vmax = 0.7,alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("Subject (sorted by bias)")
    plt.title(u'Reward allocated (gray) or won (red)')
    plt.tight_layout()
    plt.savefig(f'outputs/Sup_figure_2/A_Reward_allocated_or_won_biasP.pdf')

    # B hist of Rwards won on bias+
    plt.figure(figsize=(2,4))
    plt.hist(df_RaCaS['Detected Rewards (B+)'],bins=np.arange(0,25.01,1),weights=np.ones(len(df_RaCaS))/len(df_RaCaS),color='yellow',
                alpha=0.8,edgecolor='k',linewidth=1,label='RaCaS')
    plt.hist(df_Pool['Detected Rewards (B+)'],bins=np.arange(0,25.01,1),weights=np.ones(len(df_Pool))/len(df_Pool),color='grey',
                alpha=0.6,edgecolor='k',linewidth=1,label='Control (pooled)')
    
    plt.axvline(12.5,ls='--',color='brown')
    plt.xlabel('Rewards Won (B+)')
    plt.ylabel('Proportion of participants')
    plt.savefig(f'outputs/Sup_figure_2/B_hist_rewards_won_at_biasP.pdf')

    # C heatmap of rewards won at bias-
    plt.figure(figsize=(14,14),dpi=150)
    matrix_won_at_2 = np.array(matrix_won_at_2).astype(float)
    matrix_won_at_2[matrix_won_at_2==0] = np.nan

    plt.pcolormesh(matrix_reward_at_2,cmap=cmap_k, vmin=0.5, vmax = 0.7,alpha=0.7)

    plt.pcolormesh(matrix_won_at_2,cmap=cmap_b, vmin=0.5, vmax = 0.7,alpha=0.7)
    plt.xlabel("Trial")
    plt.ylabel("Subject (sorted by bias)")
    plt.title(u'Reward allocated (gray) or won (red)')
    plt.tight_layout()
    plt.savefig(f'outputs/Sup_figure_2/C_Reward_allocated_or_won_biasN.pdf')

    # D hist of Rwards won on bias-
    plt.figure(figsize=(2,4))
    plt.hist(df_RaCaS['Detected Rewards (B-)'],bins=np.arange(0,25.01,1),weights=np.ones(len(df_RaCaS))/len(df_RaCaS),color='yellow',
                alpha=0.8,edgecolor='k',linewidth=1,label='RaCaS')
    plt.hist(df_Pool['Detected Rewards (B-)'],bins=np.arange(0,25.01,1),weights=np.ones(len(df_Pool))/len(df_Pool),color='grey',
                alpha=0.6,edgecolor='k',linewidth=1,label='Control (pooled)')
    plt.axvline(12.5,ls='--',color='brown')
    plt.legend()
    plt.xlabel('Rewards Won (B-)')
    plt.ylabel('Proportion of participants')
    plt.savefig(f'outputs/Sup_figure_2/D_hist_rewards_won_at_biasN.pdf')

    # E boxplot of rewards won at bias+ or bias- for each schedule
    fig, axes = plt.subplots(2,1, figsize=(25,10), sharex=True, sharey=True)
    sns.boxplot(x='Model/Schedule',y='Detected Rewards (B+)',data=df_noPool_noSS00_noDS01_noDS02,ax=axes[0],
                showfliers=False,palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1))
    sns.boxplot(x='Model/Schedule',y='Detected Rewards (B-)',data=df_noPool_noSS00_noDS01_noDS02,ax=axes[1],
                showfliers=False,palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1))
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='Detected Rewards (B+)', data=df_noPool_noSS00_noDS01_noDS02,ax=axes[0],color='k',alpha=0.2)
    scatter_p(x='Model/Schedule', y='Detected Rewards (B-)', data=df_noPool_noSS00_noDS01_noDS02,ax=axes[1],color='k',alpha=0.2)
    axes[0].axhline(12.5,ls='--',color='brown')
    axes[1].axhline(12.5,ls='--',color='brown')
    plt.savefig(f'outputs/Sup_figure_2/E_boxplot_rewards_won_per_schedule.pdf')


def Sup_Figure_3(df_noPool_noSS00_noDS01_noDS02):
    # box plot of Expectancy
    os.makedirs('outputs/Sup_figure_3',exist_ok=True)
    df_noPool_noSS00_noDS01_noDS02 = df_noPool_noSS00_noDS01_noDS02.copy()
    sced_sort = df_noPool_noSS00_noDS01_noDS02.groupby('Model/Schedule')['Bias'].mean().sort_values(ascending=False).index
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].astype("category")
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].cat.set_categories(sced_sort)
    colors = ['tab:olive','tab:gray']
    fig, axes = plt.subplots(2,1,figsize=(25,10),sharex=True,sharey=True)
    sns.boxplot(x='Model/Schedule',y='Observed Expectancy (B+)',data=df_noPool_noSS00_noDS01_noDS02,ax=axes[0],
                showfliers=False,palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1))
    sns.boxplot(x='Model/Schedule',y='Observed Expectancy (B-)',data=df_noPool_noSS00_noDS01_noDS02,ax=axes[1],
                showfliers=False,palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1))
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='Observed Expectancy (B+)', data=df_noPool_noSS00_noDS01_noDS02,ax=axes[0],color='k',alpha=0.2)
    scatter_p(x='Model/Schedule', y='Observed Expectancy (B-)', data=df_noPool_noSS00_noDS01_noDS02,ax=axes[1],color='k',alpha=0.2)
    axes[0].axhline(0.25,ls='--',color='brown')
    axes[1].axhline(0.25,ls='--',color='brown')
    plt.savefig(f'outputs/Sup_figure_3/A_boxplot_expectancy_per_schedule.pdf')



def Sup_Figure_4(df_noPool_noSS00_noDS01_noDS02, df_Pool_RaCaS):
    os.makedirs('outputs/Sup_figure_4',exist_ok=True)
    df_RaCaS = df_Pool_RaCaS.query('`Model/Schedule`=="RaCaS"')
    df_Pool = df_Pool_RaCaS.query('`Model/Schedule`=="Control (pooled)"')
    df_noPool_noSS00_noDS01_noDS02 = df_noPool_noSS00_noDS01_noDS02.copy()
    sced_sort = df_noPool_noSS00_noDS01_noDS02.groupby('Model/Schedule')['Bias'].mean().sort_values(ascending=False).index
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].astype("category")
    df_noPool_noSS00_noDS01_noDS02['Model/Schedule'] = df_noPool_noSS00_noDS01_noDS02['Model/Schedule'].cat.set_categories(sced_sort)
    colors = ['tab:olive','tab:gray']

    # A box plot of rt bias of all models
    plt.figure(figsize=(25,5))
    sns.boxplot(x='Model/Schedule',y='$Δ_{RT}$:[(B+)-(B-)]',data=df_noPool_noSS00_noDS01_noDS02,ax=plt.gca(),
                palette=[colors[0]] + [colors[1]]*(len(sced_sort)-1), showfliers=False)
    scatter_p = getattr(sns,'stripplot')
    np.random.seed(123) # repeat the same stripplot
    scatter_p(x='Model/Schedule', y='$Δ_{RT}$:[(B+)-(B-)]', data=df_noPool_noSS00_noDS01_noDS02,ax=plt.gca(),color='k',alpha=0.2)
    plt.axhline(0,ls='--',color='brown')
    plt.ylim(-1.5,1.5)
    plt.savefig(f'outputs/Sup_figure_4/A_boxplot_RT_bias.pdf')

    # B rt bias vs Bias
    plt.figure(figsize=(4,4))
    plt.scatter(x=df_Pool['Bias'],y=df_Pool['$Δ_{RT}$:[(B+)-(B-)]'],color=colors[1],alpha=0.8,
                label='Control (pooled)',linewidth=0.2,edgecolor='k')
    plt.scatter(x=df_RaCaS['Bias'],y=df_RaCaS['$Δ_{RT}$:[(B+)-(B-)]'],color=colors[0],alpha=0.8,
                label='RaCaS',linewidth=0.2,edgecolor='k')

    plt.axhline(0,ls='--',color='brown')
    plt.ylim(-1.65,1.65)
    plt.savefig(f'outputs/Sup_figure_4/B_RT_bias_vs_Bias.pdf')

    # print total 
    


def plot_with_sem_winning(y,data,lw,label,ls,color):
    mn = np.nanmean(data[y],0)
    sm = sem(data[y],0,nan_policy='omit')
    plt.plot(mn,lw=lw ,label=label,ls=ls, color=color)
    plt.fill_between(range(len(sm)),mn-sm,mn+sm, color=color,alpha=0.5)

def clean_cell_3(x):
    x = x.values
    ret_text = f'µ={x[0]:.2f}, σ={x[1]:.2f}'
    p='-------'
    if x[2]<0.05:
        p='<0.05'
    if x[2]<0.01:
        p='<0.01'
    if x[2]<0.001:
        p='<1e-3'
    if x[2]<0.0001:
        p='<1e-4'
    if x[2]<0.00001:
        p='<1e-5'
    if x[2]<0.000001:
        p='<1e-6'
    if x[2]<0.0000001:
        p='<1e-7'
    if x[2]<0.00000001:
        p='<1e-8'
        
    if x[2]>=0.05:
        p=f'={x[2]:.2f}'
        
    ret_text += f', P{p}'
        
    return ret_text

def clean_cell_2(x):
    x = x.values
    ret_text = f'R={x[0]:.2f}'
    
    if x[1]<0.05:
        p='<0.05'
    if x[1]<0.01:
        p='<0.01'
    if x[1]<0.001:
        p='<1e-3'
    if x[1]<0.0001:
        p='<1e-4'
    if x[1]<0.00001:
        p='<1e-5'
    if x[1]<0.000001:
        p='<1e-6'
    if x[1]<0.0000001:
        p='<1e-7'
    if x[1]<0.00000001:
        p='<1e-8'
        
    if x[1]>=0.05:
        p=f'={x[1]:.2f}'
        
    ret_text += f', P{p}'
    return ret_text