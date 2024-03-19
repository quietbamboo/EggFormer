import argparse

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, StratifiedKFold  # 导入数据集拆分工具包
import sys, os
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

    check_dir(current_dir + '/results/PCA/')

    cv_nums = 3
    clf_importance = []
    bands_contribution = []
    bands_list = return_bands_list()
    contribution_dict = {'wavelength': bands_list}

    # 读取本地的所有光谱数据 迭代天数
    for method in ['ORI']:
        with pd.ExcelWriter(current_dir + '/results/PCA/PCA_{}_wavelength_contribution.xlsx'.format(method)) as writer:
            for days_idx in tqdm(days):
                day_input_dir = os.path.join(input_root_dir, '{}d'.format(days_idx))
                # data
                spectral_average, label_enc = return_spectral_average(day_input_dir, smooth=method)
                # count
                fold_scores = scores_save_and_average()
                skf_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
                for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
                    print('cv', cv_idx)
                    # split dataset
                    train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], spectral_average[val_idx], label_enc[val_idx]
                    print('【Test】 nums|sum: {}|{}'.format(len(test_y), test_y.sum()), '【train】 nums|sum: {}|{}'.format(len(train_y), train_y.sum()))

                    # TODO: PCA MODEL
                    pca = PCA(n_components=0.99, random_state=0)
                    pca.fit(train_x)
                    data_pca_train = pca.transform(train_x)
                    data_pca_test = pca.transform(test_x)
                    print(data_pca_test.shape)

                    pls = PLSRegression()
                    pls.fit(data_pca_train, train_y)
                    Y_train_predicted_ = [0 if _ < 0.5 else 1 for _ in pls.predict(data_pca_train)]
                    Y_test_predicted = [0 if _ < 0.5 else 1 for _ in pls.predict(data_pca_test)]

                    acc_train, prec_train, recall_train, f1_train, kappa_train, waste_ratio_train, matrix_train = output_results(train_y, Y_train_predicted_, print_flag=False)
                    test_scores = output_results(test_y, Y_test_predicted, mark=days_idx, acc_train=acc_train)
                    fold_scores.fresh_value(test_scores)

                fold_scores_mean = fold_scores.get_mean(print_flag=True)
                cv_acc_max_idx = fold_scores.print_max()
                top_dict = {}
                print('Generating best model pcs...')
                for cv_idx, (train_idx, val_idx) in enumerate(skf_val.split(spectral_average, label_enc)):
                    if cv_idx != cv_acc_max_idx:
                        continue
                    train_x, train_y, test_x, test_y = spectral_average[train_idx], label_enc[train_idx], \
                    spectral_average[val_idx], label_enc[val_idx]

                    # TODO: PCA MODEL
                    pca = PCA(n_components=0.99, random_state=0)
                    pca.fit(train_x)

                    # ratio
                    explained_variance_ratio = pca.explained_variance_ratio_
                    print("PCA自适应推荐给出的贡献率排序:", explained_variance_ratio)
                    print(explained_variance_ratio.shape)
                    components = pca.components_
                    # print('其中每个主成分对应的特征值:', components)
                    print(components.shape)
                    pd.to_pickle(components, current_dir+'/results/PCA/components_{}.pkl'.format(days_idx))

                    for pc_idx, (components_, ratio_) in enumerate(zip(components, explained_variance_ratio)):
                        indices = np.argsort(-components_)
                        wavelength_sort = np.array(bands_list)[indices]  # top wavelength
                        wavelength_contribution = components_[indices]  # top contribution

                        contribution_dict['pc{}'.format(pc_idx)] = components_

                        top_dict['wavelength_pc{}'.format(pc_idx)] = wavelength_sort
                        top_dict['contributions_pc{}'.format(pc_idx)] = wavelength_contribution
                        top_dict['idx_pc{}'.format(pc_idx)] = indices

                    pd.DataFrame({'ratio': explained_variance_ratio}).to_excel(writer, sheet_name='pc_ratio', index=False)
                    pd.DataFrame(top_dict).to_excel(writer, sheet_name='top_{}day'.format(days_idx, pc_idx), index=False)
                    pd.DataFrame(contribution_dict).to_excel(writer, sheet_name='all_range_day{}'.format(days_idx), index=False)

                    # pcs
                    save_dir = current_dir + '/results/PCA/pcs/'
                    check_dir(save_dir)
                    pca_test_img_list = ['1_male.npy', '5_male.npy', '8_male.npy', '3_female.npy', '4_female.npy', '6_female.npy']
                    for pca_test_img in pca_test_img_list:
                        PCA_list = []
                        for PC_idx in range(components.shape[0]):
                            ori_data = np.array(np.load(os.path.join(day_input_dir, pca_test_img)))
                            PC_img = ori_data[:, :, 0] * abs(components[PC_idx][0])
                            for channel_idx in range(1, ori_data.shape[2]):
                                PC_img += (np.array(ori_data[:, :, channel_idx]) * (components[PC_idx][channel_idx]))
                            _255_img = return_255_circles_ignore_padding_0(PC_img)
                            PCA_list.append(_255_img)
                            cv2.imwrite(save_dir + os.path.basename(pca_test_img).split('.')[0] + '_pc_{}.jpg'.format(PC_idx), _255_img)

                        PCA_pkl = np.array(PCA_list).transpose((2, 1, 0))
                        pd.to_pickle(np.array(PCA_pkl), save_dir + pca_test_img)

            writer.save()
