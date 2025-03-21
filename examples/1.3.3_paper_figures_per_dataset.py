import multiprocessing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.crowd_certain.utilities.utils import (
    Aim1_3_Data_Analysis_Results,
    CrowdCertainOrchestrator,
)

""" Running Calculations  """

num_seeds = 6
num_simulations = 10
low_dis, high_dis = 0.4, 1
nlabelers_list = range(3,10)


outputs = {}
for NL in tqdm( nlabelers_list, desc='looping through different # labelers' ):

    aim1_3 = CrowdCertainOrchestrator(data=data, num_simulations=num_simulations, feature_columns=feature_columns, num_labelers=NL, low_dis=low_dis, high_dis=high_dis)

    with multiprocessing.Pool(processes=num_seeds ) as pool:
        outputs[f'NL{NL}'] = pool.map( aim1_3.full_accuracy_comparison ,  list(range(num_seeds))  )



""" 2.2 Estimated weights"""
seed_ix, worker_strength_ix = 0, 2
weights = outputs[f'NL{nlabelers_list[-1]}'][seed_ix][worker_strength_ix].sort_values(by=['labelers_strength'], ascending=True).round(decimals=2)


""" 2.3 Showing the results for each seed """
A13R = Aim1_3_Data_Analysis_Results(outputs=outputs, nlabelers_list=nlabelers_list, dataset=dataset)
A13R.stacking_all_seeds()


""" 2.3 Final results - average over all seeds """
A13R.avg_accuracy_over_all_seeds(run_stacking=False)


""" Paper figures """
plt.figure(figsize=(20,5))

df_freq_stacked = A13R.accuracy['freq'].rename(columns={'uwMV-freq (proposed)':'proposed', 'uwMV-freq (proposed_penalized)':'proposed_penalized', 'wMV-freq (Tao)':'Tao', 'MV-freq (Sheng)':'Sheng'})#.reset_index()
df_freq_stacked = df_freq_stacked.stack().to_frame().reset_index().rename(columns={'level_1':'method', 0:'accuracy'})
df_freq_stacked['strategy'] = 'freq'

df_beta_stacked = A13R.accuracy['beta'].rename(columns={'uwMV-beta (proposed)':'proposed', 'uwMV-beta (proposed_penalized)':'proposed_penalized', 'wMV-beta (Tao)':'Tao', 'MV-beta (Sheng)':'Sheng'})#.reset_index()
df_beta_stacked = df_beta_stacked.stack().to_frame().reset_index().rename(columns={'level_1':'method', 0:'accuracy'})
df_beta_stacked['strategy'] = 'beta'

df = pd.concat([df_freq_stacked, df_beta_stacked], axis=0)
df.head()



""" 3.1 Comparing the proposed METHODS """
fig = plt.figure(figsize=(20,5))

plt.subplot(121)
sns.violinplot(x='strategy', y='accuracy', data=df[df.method=='proposed'])
plt.title('proposed')

plt.subplot(122)
p = sns.violinplot(x='strategy', y='accuracy', data=df[df.method=='proposed_penalized'])
p.set_title('proposed_penalized')

path = f'figures/Proposed method comparison - Dataset {dataset}.png'
fig.savefig(path, dpi=300)



""" Estimated-weight vs Worker-strength """
aim1_3 = Aim1_3_Data_Analysis_Results(nlabelers_list=nlabelers_list, dataset=dataset)
_, df_comparison_Tao_stacked = aim1_3.worker_weight_strength_relation(smooth=True, seed=2, data=data, num_simulations=num_simulations, interpolation_pt_count=1000, feature_columns=feature_columns, num_labelers=20, low_dis=0.4, high_dis=1)

p = sns.jointplot(data=df_comparison_Tao_stacked, x="worker strength", y="measured weight", hue="method",  ylim=(0,1.6), xlim=(0,1), kind='scatter', joint_kws={"s": 1}, ratio=3, size=7, space=0.1)
p.ax_joint.plot(aim1_3.weight_strength_relation[ ['proposed_penalized', 'Tao'] ], 'o')
p.ax_marg_x.set_title(f'Dataset: {dataset}')
p.ax_joint.legend(loc='lower right')

path = f'figures/Estimated-weight vs Worker-strength - Dataset {dataset} - via seaborn.png'
p.savefig(path, dpi=300)



""" The accuracy distribution using kernel density function """
fig = plt.figure(figsize=(20,5))

for i, strategy in enumerate(['freq','beta']):

    plt.subplot(1,2,i+1)
    rename_dict = {f'uwMV-{strategy} (proposed_penalized)':'proposed_penalized', f'wMV-{strategy} (Tao)':'Tao', f'MV-{strategy} (Sheng)':'Sheng'} # , 'MajorityVote':'MV'

    df_comparison_Tao = A13R.accuracy[strategy][rename_dict.keys()].rename(columns=rename_dict)

    sns.kdeplot(data=df_comparison_Tao, shade=True, legend=True, cbar=True)
    plt.xlabel('accuracy')
    plt.title(strategy.capitalize())

path = f'figures/Accuracy distribution comparison to Tao and Sheng - Dataset {dataset}.jpg'
fig.savefig(path, dpi=300)


"""  figure shows the average accuracies across different technigues for different # of workers """
for strategy in ['freq', 'beta']:

    fig = plt.figure(figsize=(20,7))
    df = A13R.accuracy[strategy].rename(columns={f'uwMV-{strategy} (proposed)':'proposed', f'uwMV-{strategy} (proposed_penalized)':'proposed_penalized', f'wMV-{strategy} (Tao)':'Tao', f'MV-{strategy} (Sheng)':'Sheng'}).drop(columns=['MV_Classifier'])

    sns.heatmap(df.iloc[:6], annot=True, fmt='.2f', cmap='Blues', cbar=True, robust=True)
    plt.title(f'average accuracy - Dataset {dataset}  - {strategy.upper()} \n')
    plt.ylabel('# workers')


    path = f'figures/Proposed method vs all benchmarks  average accuracy - Dataset {dataset} - {strategy}.jpg'
    fig.savefig(path, dpi=300)



""" This figure shows the distribution of average accuracies across different technigues for # workers smaller than 6  """
for strategy in ['freq', 'beta']:

    fig = plt.figure(figsize=(20,7))
    BOUNDARY = max(nlabelers_list)
    data = eval(f'df_{strategy}_stacked')
    sns.violinplot(x='method', y='accuracy', data=data[data.nlabelers <= BOUNDARY])
    plt.title(f'average accuracy distribution    -     # workers <= {BOUNDARY} - {strategy.upper()}')


    path = f'figures/Proposed method vs all benchmarks density function - Dataset {dataset} - {strategy}.jpg'
    fig.savefig(path, dpi=600)


""" 3.1 Comparing the METHOD 1 & 2 """
fig = plt.figure(figsize=(20,5))
A13R.plot_comparing_proposed_methods_1_2(smooth=True)

path = f'figures/Proposed method 1 vs 2 - Dataset {dataset}.jpg'
fig.savefig(path, dpi=300)



""" 4.1 Comparing the FREQ & BETA strategies """
fig = plt.figure(figsize=(20,5))
A13R.plot_comparing_proposed_methods_freq_beta(smooth=True)

path = f'figures/Proposed method freq vs beta - Dataset {dataset}.jpg'
fig.savefig(path, dpi=300)



""" 4.2 Comparing with Tao & Sheng """
fig = plt.figure(figsize=(20,5))
A13R.plot_comparing_proposed_with_Tao_Sheng_MV(smooth=True)

path = f'figures/Proposed method comparison to Tao and Sheng - Dataset {dataset}.jpg'
fig.savefig(path, dpi=300)



""" 4.3 Comparing with all benchmarks """
for strategy in ['freq', 'beta']:

    fig = plt.figure(figsize=(14,5))
    A13R.plot_comparing_proposed_with_all_benchmarks(strategy=strategy, smooth=True, legend={'loc':'upper left', 'bbox_to_anchor':(1,1)}, title=f'Dataset: {dataset}   -   {strategy.upper()}')

    path = f'figures/Proposed method comparison to all benchmarks - Dataset {dataset} - {strategy.upper()}.jpg'
    fig.savefig(path, dpi=300)
