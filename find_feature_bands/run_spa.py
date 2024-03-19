"""从npy文件调用数据+平均光谱+ SPA + PLSDA 提取 significant wavelength"""
import argparse

from sklearn.model_selection import train_test_split, StratifiedKFold  # 导入数据集拆分工具包
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
import sys, os
from spa_functions import SPA
sys.path.insert(0, sys.path[0]+"/../")
current_dir = os.path.dirname(sys.argv[0])
from utils import *


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input-root-dir', default='/nfs/my/Xu/jicm/food_datasets_npy')
    args.add_argument('--days', default=[10])
    opt = args.parse_args()

    days = opt.days
    input_root_dir = opt.input_root_dir

    save_dir = current_dir + '/results/SPA/'
    min_num = 10
    max_num = 60
    check_dir(save_dir)
    cv_nums = 3
    clf_importance = []
    bands_contribution = []
    bands_list = np.array(return_bands_list())

    # 读取本地的所有光谱数据 迭代天数
    pd_spa_results = {}
    for method in ['ORI']:
        for days_idx in tqdm(days):
            # data
            spectral_average, label_enc = return_spectral_average(os.path.join(input_root_dir, '{}d'.format(days_idx)), smooth=method)
            # count
            fold_scores = scores_save_and_average()
            skf_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
                print('cv', cv_idx)
                # split dataset
                train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], spectral_average[val_idx], label_enc[val_idx]
                print('【Test】 nums|sum: {}|{}'.format(len(test_y), test_y.sum()), '【train】 nums|sum: {}|{}'.format(len(train_y), train_y.sum()))
                # TODO: SPA MODEL
                var_sel, var_sel_phase2, relev, index_decreasing_relev, RMSEP_scree = SPA().spa(train_x, train_y, m_min=min_num, m_max=max_num, Xval=test_x, yval=test_y, autoscaling=1)

                x_spa_train = np.array(train_x)[:, var_sel]
                y_spa_train = train_y
                x_spa_test = np.array(test_x)[:, var_sel]
                y_spa_test = test_y

                # 调用模型计算test集合
                pls = PLSRegression()
                pls.fit(x_spa_train, train_y)
                Y_train_predicted_ = [0 if _ < 0.5 else 1 for _ in pls.predict(x_spa_train)]
                Y_test_predicted = [0 if _ < 0.5 else 1 for _ in pls.predict(x_spa_test)]

                acc_train, prec_train, recall_train, f1_train, kappa_train, waste_ratio_train, matrix_train = output_results(y_spa_train, Y_train_predicted_, print_flag=False)
                test_scores = output_results(y_spa_test, Y_test_predicted, mark=days_idx, acc_train=acc_train)
                fold_scores.fresh_value(test_scores)

            # TODO 统计结果
            fold_scores_mean = fold_scores.get_mean(print_flag=True)
            cv_acc_max_idx = fold_scores.print_max()
            top_dict = {}
            print('Generating best model pcs...')
            for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
                if cv_idx != cv_acc_max_idx:
                    continue

            # split dataset
            train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], spectral_average[ val_idx], label_enc[val_idx]
            print('【Test】 nums|sum: {}|{}'.format(len(test_y), test_y.sum()), '【train】 nums|sum: {}|{}'.format(len(train_y), train_y.sum()))
            # TODO: SPA MODEL
            var_sel, var_sel_phase2, relev, index_decreasing_relev, RMSEP_scree = SPA().spa(train_x, train_y, m_min=min_num, m_max=max_num, Xval=test_x, yval=test_y, autoscaling=1)

            x_spa_train = np.array(train_x)[:, var_sel]
            y_spa_train = train_y
            x_spa_test = np.array(test_x)[:, var_sel]
            y_spa_test = test_y

            # 调用模型计算test集合
            pls = PLSRegression()
            pls.fit(x_spa_train, train_y)
            Y_train_predicted_ = [0 if _ < 0.5 else 1 for _ in pls.predict(x_spa_train)]
            Y_test_predicted = [0 if _ < 0.5 else 1 for _ in pls.predict(x_spa_test)]

            acc_train, prec_train, recall_train, f1_train, kappa_train, waste_ratio_train, matrix_train = output_results(
                y_spa_train, Y_train_predicted_, print_flag=False)
            test_scores = output_results(y_spa_test, Y_test_predicted, mark=days_idx, acc_train=acc_train)

            pd_spa_results['{}d_selected_ws'.format(days_idx)] = [bands_list[var_sel]]
            pd_spa_results['{}d_acc'.format(days_idx)] = [test_scores[0]]
            pd_spa_results['{}d_recall'.format(days_idx)] = [test_scores[2]]
            pd_spa_results['{}d_spectrum'.format(days_idx)] = [train_x[0]]
            pd_spa_results['{}d_candidate_ws'.format(days_idx)] = [bands_list[var_sel_phase2]]
            pd_spa_results['{}d_sorted_relevance'.format(days_idx)] = [relev[index_decreasing_relev]]
            pd_spa_results['{}d_RMSEP_scree'.format(days_idx)] = [RMSEP_scree]

    pd.DataFrame(pd_spa_results).to_excel(save_dir + '/SPA_ORI_wavelength_contribution.xlsx', index=False)
    pd.DataFrame(pd_spa_results).to_pickle(save_dir + '/SPA_ORI_wavelength_contribution.pkl')

