"""从npy文件调用数据+平均光谱+预处理函数 + RandomForest"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score  # 导入数据集拆分工具包
import sys, os
sys.path.insert(0, sys.path[0]+"/../")
current_dir = os.path.dirname(sys.argv[0])
from utils import *
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input-root-dir', default='/nfs/my/Xu/jicm/food_datasets_npy')
    args.add_argument('--days', default=[0, 2, 4, 6, 8, 10, 12, 14])
    opt = args.parse_args()

    days = opt.days
    input_root_dir = opt.input_root_dir

    check_dir(current_dir + '/results/')
    clf_importance = []
    importance = 0
    cv_nums = 3
    score_names = ['accuracy', 'precision', 'recall', 'f1', 'kappa']
    methods = ['ORI', 'SG', 'SNV', 'MSC', 'D1', 'D2']
    mean_scores = {name_ + '_mean': deepcopy([]) for name_ in score_names}
    method_scores = {method_name_: deepcopy(mean_scores) for method_name_ in methods}

    # 读取本地的所有光谱数据 迭代天数
    for days_idx in tqdm(days):
        # data
        spectral_average_ori, label_enc = return_spectral_average(os.path.join(input_root_dir, '{}d'.format(days_idx)), smooth='ORI')
        score_aas, score_oas, score_recall, score_f1, score_kappa, score_matrix = [], [], [], [], [], []

        for method in methods:
            if method == 'ORI':
                spectrum = spectral_average_ori
            elif method == 'SG':
                spectrum = return_sg_spectrum(spectral_average_ori)
            elif method == 'SNV':
                spectrum = return_snv_spectrum(spectral_average_ori)
            elif method == 'MSC':
                spectrum = return_msc_spectrum(spectral_average_ori)
            elif method == 'D1':
                spectrum = return_d1_spectrum(spectral_average_ori)
            elif method == 'D2':
                spectrum = return_d2_spectrum(spectral_average_ori)

            fold_scores = scores_save_and_average()
            score_dict = {name_: deepcopy([]) for name_ in score_names}
            skf_val = StratifiedKFold(n_splits=cv_nums, shuffle=True, random_state=0)
            for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectrum, label_enc)):
                print('cv', cv_idx)
                # split dataset
                train_x, train_y, test_x, test_y = spectrum[train_idx], label_enc[train_idx], spectrum[val_idx], label_enc[val_idx]

                # model
                clf = RandomForestClassifier(random_state=0)
                clf.fit(train_x, train_y)
                predicted = clf.predict(test_x)
                scores = output_results(test_y, predicted)

                for name_, score_ in zip(score_names, scores):
                    score_dict[name_].append(score_)

                print(f"【Day {days_idx:d} method {method}】Test acc {scores[0]:.3f} prec {scores[1]:.3f} recall {scores[2]:.3f} f1 {scores[3]:.3f} kappa {scores[4]:.3f}")

            for name_ in score_names:
                method_scores[method][name_+'_mean'].append(np.array(score_dict[name_]).mean())

    for method_name_ in methods:
        method_scores[method_name_]['days'] = days
        df = method_scores[method_name_]
        pd.DataFrame(df).to_excel(current_dir + '/results/method_{}.xlsx'.format(method_name_), index=False)

