import argparse
import sys, os
from sklearn.cross_decomposition import PLSRegression
sys.path.insert(0, sys.path[0]+"/../")
current_dir = os.path.dirname(sys.argv[0])
from sklearn.model_selection import StratifiedKFold  # 导入数据集拆分工具包
from utils import *
from scipy.io import savemat
from scipy.io import loadmat


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input-root-dir', default='/nfs/my/Xu/jicm/food_datasets_npy')
    args.add_argument('--days', default=[10])
    opt = args.parse_args()

    days = opt.days
    input_root_dir = opt.input_root_dir

    save_dir = current_dir + '/results/CARS/'
    check_dir(save_dir)
    method = 'ORI'
    spectral_average, label_enc = return_spectral_average(os.path.join(input_root_dir, '{}d'.format(days[0])), smooth=method)
    skf_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    mat_files = ['CARS_CV0.mat', 'CARS_CV1.mat', 'CARS_CV2.mat']

    fold_scores = scores_save_and_average()
    save_mat_flag = True

    for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
        train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], spectral_average[val_idx], label_enc[val_idx]
        print('【Test】 nums|sum: {}|{}'.format(len(test_y), test_y.sum()), '【train】 nums|sum: {}|{}'.format(len(train_y), train_y.sum()))

        if save_mat_flag:
            # TODO generate CARS input mat files
            mat_dict = {'Xcal': np.array(train_x), 'ycal': np.array(train_y).astype(np.float).reshape(-1,1)}
            savemat(save_dir + 'D2_8day_train_cv{}.mat'.format(cv_idx), mat_dict)

            mat_dict = {'Xval': np.array(test_x), 'yval': np.array(test_y).astype(np.float).reshape(-1,1)}
            savemat(save_dir + 'D2_8day_val_cv{}.mat'.format(cv_idx), mat_dict)
        else:
            mat_file_name = mat_files[cv_idx]
            input_mat_file_path = os.path.join(save_dir, mat_file_name)
            mat_data = loadmat(input_mat_file_path)['CARS1']['vsel']
            var_sel = mat_data[0][0].flatten()

            x_cars_train = np.array(train_x)[:, var_sel]
            y_cars_train = train_y
            x_spa_test = np.array(test_x)[:, var_sel]
            y_spa_test = test_y

            # 调用模型计算test集合
            pls = PLSRegression()
            pls.fit(x_cars_train, train_y)
            Y_train_predicted_ = [0 if _ < 0.5 else 1 for _ in pls.predict(x_cars_train)]
            Y_test_predicted = [0 if _ < 0.5 else 1 for _ in pls.predict(x_spa_test)]

            acc_train, prec_train, recall_train, f1_train, kappa_train, waste_ratio_train, matrix_train = output_results(y_cars_train, Y_train_predicted_, print_flag=False)
            test_scores = output_results(y_spa_test, Y_test_predicted, mark=10, acc_train=acc_train)
            fold_scores.fresh_value(test_scores)

    if not save_mat_flag:
        # TODO 统计结果
        fold_scores_mean = fold_scores.get_mean(print_flag=True)
        cv_acc_max_idx = fold_scores.print_max()
