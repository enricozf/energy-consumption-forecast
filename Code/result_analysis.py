#%%
# Imports
import joblib
import matplotlib.pyplot as plt

import optuna

# %matplotlib inline
#%%
if __name__ == '__main__':
    result_filepath = '..//Results//{}.pkl'
    dic_models = {
        0 : 'dense_many2many_v0',
        1 : 'dense_many2one_v0',
        2 : 'dense_many2one_v1',
        3 : 'mlp_mixer_many2one_v0',
        4 : 'lstm_many2one_v0',
        5 : 'lstm_many2one_v1',
        6 : 'lstm_many2one_v2',
        7 : 'conv1d_lstm_many2one_v0',
        8 : 'conv1d_lstm_many2one_v1',
        9 : 'conv1d_lstm_many2one_v2',
        10 : 'conv1d_lstm_many2one_v3',
        11 : 'wavenet_many2one_v0',
        12 : 'wavenet_many2one_v1',
        13 : 'wavenet_mlp_mixer_many2one_v0',
        14 : 'wavenet_mlp_mixer_many2one_v1'
    }

    # for model in dic_models.values():
    #     try: 
    #         with open(result_filepath.format(model), 'rb') as f:
    #             results = joblib.load(f)
    #             print('Using {} model:'.format(model))
    #             print(results.best_trial)
    #             print()
    #     except FileNotFoundError:
    #         continue
    font_size=20
    with open(result_filepath.format(dic_models[10]), 'rb') as f:
        results = joblib.load(f)
    fig = optuna.visualization.plot_parallel_coordinate(results,
                                                        # params=['1st_layer','2nd_layer'],
                                                        target_name='MSE')
    fig.update_layout(
        title='', font={'size':font_size}
    )
    fig.show()

    fig = optuna.visualization.plot_optimization_history(results,
                                                         target_name='MSE')
    fig.update_layout(
        title='', font={'size':font_size}
    )
    fig.show()

#%%
from utils.data_prep import gen_dataset

dic_split, scaler = gen_dataset(
    fold_json_path='..//Data//folds.json',
    time_col='index', scale_flg=True,
    num_sm_split = {
        'train':2, 
        'val':1, 
        'test':1
    },
    boxcox_trnsf_flag=True)