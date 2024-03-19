import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import shap
import sys, os
sys.path.insert(0, sys.path[0] + "/../")
current_dir = os.path.dirname(sys.argv[0])
from utils import *


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input-root-dir', default='/nfs/my/Xu/jicm/food_datasets_npy')
    args.add_argument('--days', default=[10])
    opt = args.parse_args()

    days = opt.days
    input_root_dir = opt.input_root_dir

    check_dir(current_dir + '/results/RF/')

    cv_nums = 3
    clf_importance = []
    bands_contribution = []
    bands_list = return_bands_list()
    contribution_dict = {'wavelength': bands_list}

    # 读取本地的所有光谱数据 迭代天数
    for method in ['ORI']:
        with pd.ExcelWriter(current_dir + '/results/Shapley_{}_wavelength_contribution.xlsx'.format(method)) as writer:
            for days_idx in tqdm(days):
                # data
                spectral_average, label_enc = return_spectral_average(os.path.join(input_root_dir, '{}d'.format(days_idx)), smooth=method)

                fold_scores = scores_save_and_average()
                skf_val = StratifiedKFold(n_splits=cv_nums, shuffle=True, random_state=0)
                for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
                    print('cv', cv_idx)
                    # split dataset
                    train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], spectral_average[val_idx], label_enc[val_idx]
                    print('【Test】 nums|sum: {}|{}'.format(len(test_y), test_y.sum()), '【train】 nums|sum: {}|{}'.format(len(train_y), train_y.sum()))

                    # model
                    clf = RandomForestClassifier(random_state=0)
                    clf.fit(train_x, train_y)

                    # results
                    test_y_predicted = clf.predict(test_x)
                    train_scores = output_results(train_y, clf.predict(train_x), print_flag=False)
                    test_scores = output_results(test_y, test_y_predicted)
                    print('acc: ', test_scores[0], 'matrix: ', test_scores[-1])
                    fold_scores.fresh_value(test_scores)

                fold_scores_mean = fold_scores.get_mean(print_flag=True)
                cv_acc_max_idx = fold_scores.print_max()
                top_dict = {}
                for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
                    if cv_idx == cv_acc_max_idx:
                        break

                train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], spectral_average[val_idx], label_enc[val_idx]
                # model
                clf = RandomForestClassifier(random_state=0)
                clf.fit(train_x, train_y)

                # todo RF 输出并保存特征重要性
                importance = clf.feature_importances_
                # 贡献值排序
                indices = np.argsort(-importance)
                wavelength_sort = np.array(bands_list)[indices]  # top wavelength
                wavelength_contribution = importance[indices]  # top contribution

                contribution_count = 0
                for idx_, con_ in enumerate(wavelength_contribution):
                    contribution_count += con_
                    if contribution_count > 0.99:
                        break

                top_wavelengths = wavelength_sort[: idx_]
                top_contributions = wavelength_contribution[: idx_]
                top_idx = indices[: idx_]

                # all bands
                contribution_dict['{}_day_RF'.format(days_idx)] = importance

                # top bands
                top_dict['wavelength'] = top_wavelengths
                top_dict['contributions'] = top_contributions
                top_dict['idx'] = top_idx
                pd.DataFrame(top_dict).to_excel(writer, sheet_name='top_{}day_RF'.format(days_idx), index=False)

                # TODO: Shapley
                explainer = shap.TreeExplainer(clf, link='logit')
                shap_values_all = explainer.shap_values(train_x)

                plt.figure(dpi=300)
                # shap.summary_plot(shap_values_all, train_x, plot_type='bar', feature_names=bands_list)
                shap.summary_plot(shap_values_all[1], train_x, feature_names=[round(band_, 2) for band_ in bands_list],
                                  color_bar_label='$Feature$  $Value$', color=plt.get_cmap("cool"),
                                  show=False, max_display=10)
                plt.savefig(current_dir + '/results/RF/Shapley_{}d.png'.format(days_idx))
                plt.clf()

                # todo Shap 输出并保存特征重要性
                shap_importance = np.array(shap_values_all[1]).mean(axis=0)

                # 贡献值排序
                indices = np.argsort(-shap_importance)
                wavelength_sort = np.array(bands_list)[indices]  # top wavelength
                wavelength_contribution = shap_importance[indices]  # top contribution

                # all bands
                contribution_dict['{}_day_Shap'.format(days_idx)] = shap_importance

                # top bands
                top_dict['wavelength'] = wavelength_sort
                top_dict['contributions'] = wavelength_contribution
                top_dict['idx'] = indices
                pd.DataFrame(top_dict).to_excel(writer, sheet_name='top_{}day_Shapley'.format(days_idx), index=False)

            pd.DataFrame(contribution_dict).to_excel(writer, sheet_name='all_range', index=False)
            writer.save()

