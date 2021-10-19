#%%
from utils.final_test_scripts import (
    conv1d_lstm_v3_m2o_final_test, conv1d_lstm_v3_m2o_final_test_not_true)

if __name__ == '__main__':
    for std_thrs in [4, 5, 6, 8]:
        conv1d_lstm_v3_m2o_final_test(
            folds_num=['3'],
            std_outlier_thrs=std_thrs,
            result_json_path='..//Results//results_fold_3_thrs_{}.json'
        )
        # conv1d_lstm_v3_m2o_final_test_not_true(
        #     std_outlier_thrs=std_thrs,
        #     result_json_path='..//Results//results_fold_3_thrs_{}.json'
        # )