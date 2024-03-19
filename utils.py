import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys
import cv2
from sklearn.metrics import f1_score, recall_score, confusion_matrix, accuracy_score, precision_score, cohen_kappa_score
from einops import rearrange, repeat
import torch.utils.data as Data
import torch
from copy import deepcopy
from scipy.signal import savgol_filter
import numba
import random
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat
import torch.nn as nn
import platform
from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomErasing,
    RandomRotation,
    RandomGaussianBlur,
)


warnings.filterwarnings("ignore")
current_dir = os.path.dirname(sys.argv[0])
encoder_dict = {'male': 1, 'female': 0}


# 用于编码female 和male
def encoder_label(input_label):
    output_array = []
    for label in input_label:
        _array = encoder_dict[label]
        output_array.append(_array)

    return output_array


class DataAugmentation(nn.Module):
    def __init__(self, p=0.8):
        super().__init__()
        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=p),
            RandomGaussianBlur((3, 3), (0.1, 5.0), p=p),
            RandomRotation(p=p, degrees=180.0),
            RandomErasing(scale=(0.02, 0.05), ratio=(0.3, 3.3), p=p),
        )

    @torch.no_grad()  # 禁用梯度以提高效率
    def forward(self, x):
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


# 检查文件夹路径是否存在，不存在则创建
def check_dir(check_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)


@numba.jit()
def resize_imgs(data):
    data = data.transpose((2, 1, 0))
    return data


def return_dataset(dataset_dir_path, dsize):
    print('Reading original datasets...')
    all_data, all_label = [], []
    for file_name in tqdm(os.listdir(dataset_dir_path)):
        if not file_name.endswith('.npy'):
            continue

        file_path = os.path.join(dataset_dir_path, file_name)
        data = np.load(file_path)

        data = cv2.resize(data, dsize=dsize)

        all_data.append(resize_imgs(data))
        all_label.append(file_name.split('.npy')[0].split('_')[1])

    # for data in all_data:
    dataset_x = all_data
    dataset_y = all_label

    dataset_x = np.array(dataset_x).astype(np.float32)
    dataset_y = np.array(encoder_label(dataset_y))

    return dataset_x, dataset_y


def return_dataset_pkl(dataset_dir_path):
    print('Reading pkl datasets...')
    all_data, all_label = [], []
    for file_name in tqdm(os.listdir(dataset_dir_path)):
        if not file_name.endswith('.pkl'):
            continue

        file_path = os.path.join(dataset_dir_path, file_name)
        data = pd.read_pickle(file_path)

        all_data.append(data)
        all_label.append(file_name.split('.pkl')[0].split('_')[1])

    # for data in all_data:
    dataset_x = all_data
    dataset_y = all_label

    dataset_x = np.array(dataset_x).astype(np.float32)
    dataset_y = np.array(encoder_label(dataset_y))

    return dataset_x, dataset_y

def return_dataset_with_wavelength_idx(dataset_dir_path, dsize, wavelength_idx=False):
    if not wavelength_idx:
        dataset_x, dataset_y = return_dataset(dataset_dir_path, dsize)
    else:
        print('Reading original datasets with idx {}...'.format(wavelength_idx))
        all_data, all_label = [], []
        for file_name in tqdm(os.listdir(dataset_dir_path)):
            if not file_name.endswith('.npy'):
                continue

            file_path = os.path.join(dataset_dir_path, file_name)

            if wavelength_idx != 'pca_img':
                data = np.load(file_path)[:, :, wavelength_idx]
            else:
                data = return_bands_PCA_img(np.load(file_path))
            data = cv2.resize(data, dsize=dsize)

            all_data.append(resize_imgs(data))
            all_label.append(file_name.split('.npy')[0].split('_')[1])

        # for data in all_data:
        dataset_x = all_data
        dataset_y = all_label
        dataset_x = np.array(dataset_x).astype(np.float32)
        dataset_y = np.array(encoder_label(dataset_y))
    print('\nInput dataset wavelengths numbers: ', len(dataset_x[0]))
    return dataset_x, dataset_y


def return_data_loader(x, y, batch_size, shuffle=True, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    label_loader = Data.DataLoader(Data.TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    return label_loader


def count_npy_nums(input_dir_path):
    count_num = 0
    for file_name in os.listdir(input_dir_path):
        if file_name.endswith('.npy'):
            count_num += 1

    return count_num


def return_npy_paths(input_dir_path):
    file_path_list = []
    for file_name in os.listdir(input_dir_path):
        if file_name.endswith('.npy'):
            file_path_list.append(input_dir_path + '/' + file_name)

    return file_path_list


def return_bands_list(xls_path=False):
    if not xls_path:
        df = pd.read_excel(current_dir + '/../xls/bands.xlsx')
    else:
        df = pd.read_excel(xls_path)
    bands = df['bands'].values.tolist()
    return bands


def return_path_labels(file_dir):
    file_names = os.listdir(file_dir)
    # Y label
    label_names = [_.split('_')[1].split('.npy')[0] for _ in file_names]

    label_enc = encoder_label(label_names)

    # X data
    file_paths = [os.path.join(file_dir, _) for _ in file_names]

    return file_paths, label_enc


# """返回平均光谱信息"""
def return_spectral_average(file_dir, smooth=False):
    file_paths, label_enc = return_path_labels(file_dir)
    spectral_average_all = []
    for file_path in tqdm(file_paths):
        average_ = []
        spectral_value = np.load(file_path)
        for i in range(spectral_value.shape[2]):
            data_ = spectral_value[:, :, i]
            value_, R_ = return_average_value(data_)
            average_.append(value_)
        # spectral_average_all.append(np.array(average_)[random_sequence])
        spectral_average_all.append(average_)

    spectral_average_all = np.array(spectral_average_all)
    if smooth != 'ORI':
        if smooth == 'SG':
            spectral_average_all = return_sg_spectrum(spectral_average_all)
        if smooth == 'SNV':
            spectral_average_all = return_snv_spectrum(spectral_average_all)
        if smooth == 'MSC':
            spectral_average_all = return_msc_spectrum(spectral_average_all)
        if smooth == 'D1':
            spectral_average_all = return_d1_spectrum(spectral_average_all)
        if smooth == 'D2':
            spectral_average_all = return_d2_spectrum(spectral_average_all)

    return np.array(spectral_average_all), np.array(label_enc)


@numba.jit(nopython=True)
def return_average_value(image, calculate_min_percent=0.0, calculate_max_percent=1.0):
    row_min = 0
    row_max = image.shape[0]
    for row in range(len(image)):
        for col in range(len(image[0])):
            if image[row][col] != 0:
                if row_min == 0:
                    row_min = row
                else:
                    row_max = row
    R2 = row_max - row_min
    R = np.ceil(R2 / 2)

    mid_x, mid_y = row_max / 2, row_max / 2

    value = 0
    count = 0
    for row in range(len(image)):
        for col in range(len(image[0])):
            if np.sqrt((row - mid_x) ** 2 + (col - mid_y) ** 2) <= R * calculate_max_percent and \
                    np.sqrt((row - mid_x) ** 2 + (col - mid_y) ** 2) >= R * calculate_min_percent:
                value += image[row][col]
                count += 1
    average_ = value / count

    return average_, R


# """ 输入格式为 nums * features """
def return_msc_spectrum(spectrum):
    spectrum_nums = spectrum.shape[0]
    k = np.zeros(spectrum_nums)
    b = np.zeros(spectrum_nums)

    mean = np.array(np.mean(spectrum, axis=0))
    from sklearn.linear_model import LinearRegression

    for idx_i in range(spectrum_nums):
        features = spectrum[idx_i, :]
        features = features.reshape(-1, 1)
        mean = mean.reshape(-1, 1)
        model = LinearRegression()
        model.fit(mean, features)
        k[idx_i] = model.coef_
        b[idx_i] = model.intercept_

    msc_data = np.zeros_like(spectrum)
    for idx_i in range(spectrum_nums):
        bb = np.repeat(b[idx_i], spectrum.shape[1])
        kk = np.repeat(k[idx_i], spectrum.shape[1])
        temp = (spectrum[idx_i, :] - bb) / kk
        msc_data[idx_i, :] = temp

    return msc_data


def return_snv_spectrum(spectrum):
    spectrum_nums = spectrum.shape[0]
    feature_nums = spectrum.shape[1]

    snv_data = deepcopy(spectrum)
    mean = np.mean(snv_data, axis=1)
    temp1 = np.tile(mean, snv_data.shape[1]).reshape((spectrum_nums, feature_nums), order='F')
    std = np.std(snv_data, axis=1)  # 标准差
    temp2 = np.tile(std, snv_data.shape[1]).reshape((spectrum_nums, feature_nums), order='F')

    return (snv_data - temp1) / temp2


def return_sg_spectrum(spectrum, w=5, p=3, d=0):
    sg_data = deepcopy(spectrum)
    sg_data = savgol_filter(sg_data, w, polyorder=p, deriv=d)

    return sg_data


def return_d1_spectrum(spectrum):
    d1_data = deepcopy(spectrum)

    df_d1 = pd.DataFrame(d1_data)
    df_d1_data = df_d1.diff(axis=1)  # 对横轴的数据求导
    return np.delete(df_d1_data.values, 0, axis=1)


def return_d2_spectrum(spectrum):
    d1_data = return_d1_spectrum(spectrum)

    d2_data = (pd.DataFrame(d1_data)).diff(axis=1)
    return np.delete(d2_data.values, 0, axis=1)


def return_idx_of_wavelength(input_wavelength):
    wavelength_xlsx_path = current_dir + '/../xls/bands.xlsx'
    wavelength_df = pd.read_excel(wavelength_xlsx_path)
    wavelength_columns = wavelength_df.iloc[:, 0].values.tolist()
    for idx, wavelength_ in enumerate(wavelength_columns):
        if float(wavelength_) < input_wavelength:
            continue
        else:
            return idx, wavelength_
    return -1, -1


def return_bands_RF():
    df = pd.read_excel(current_dir + '/../xls/significant_wavelengths/Shapley_ORI_wavelength_contribution.xlsx',
                       sheet_name='top_10day_RF')
    idxs = []
    for row_idx in range(len(df)):
        contributions = df.iloc[row_idx, 1]
        # if contributions >= 1e-3:
        idxs.append(df.iloc[row_idx, 2])

    return idxs


def return_bands_SPA():
    df = pd.read_pickle(current_dir + '/../xls/significant_wavelengths/SPA_ORI_wavelength_contribution.pkl')
    selected_ws = df['10d_selected_ws'][0]
    # selected_ws = selected_ws.replace('\n', '').replace('[', '').replace(']', '').split(' ')
    all_bands = return_bands_list(current_dir + '/../xls/bands.xlsx')
    selected_ws_idxs = [all_bands.index(float(ws)) for ws in selected_ws]

    return selected_ws_idxs


def return_bands_PCA_img(input_X):
    components = pd.read_pickle(current_dir + '/../xls/significant_wavelengths/components_10.pkl')

    PCA_list = []
    for PC_idx in range(components.shape[0]):
        ori_data = input_X
        PC_img = ori_data[:, :, 0] * (components[PC_idx][0])
        # PC_img = ori_data[:, :, 0] * abs(components[PC_idx][0])
        for channel_idx in range(1, ori_data.shape[2]):
            PC_img += (np.array(ori_data[:, :, channel_idx]) * (components[PC_idx][channel_idx]))
        _255_img = return_255_circles_ignore_padding_0(PC_img)/255.0
        PCA_list.append(_255_img)

    PCs = np.array(PCA_list).transpose((2, 1, 0))

    return PCs


def return_bands_PCA_channel():
    components = pd.read_pickle(current_dir + '/../xls/significant_wavelengths/components_10.pkl')

    PC_idx = 0

    eigenvectors = np.array(components[PC_idx])
    eigenvectors_abs = np.absolute(eigenvectors)
    eigenvectors_abs_sort = np.argsort(eigenvectors_abs*(-1))

    eigenvectors_abs_normal = eigenvectors_abs / np.sum(eigenvectors_abs)

    idx_list = []
    account = 0
    for idx in eigenvectors_abs_sort:
        if account + eigenvectors_abs_normal[idx] < 0.98:
            idx_list.append(idx)
            account += eigenvectors_abs_normal[idx]
        else:
            break

    return idx_list


def return_bands_CARS():
    mat_data = loadmat(current_dir + '/../xls/significant_wavelengths/CARS_CV0.mat')['CARS1']['vsel']
    var_sel = mat_data[0][0].flatten().tolist()

    return var_sel


@numba.jit(nopython=True)
def return_255_circles_ignore_padding_0(image):
    new_image = np.copy(image)
    row_min = 0
    row_max = image.shape[0]
    for row in range(len(image)):
        for col in range(len(image[0])):
            if image[row][col] != 0:
                if row_min == 0:
                    row_min = row
                else:
                    row_max = row
    R2 = row_max - row_min
    R = np.ceil(R2 / 2)

    mid_x, mid_y = row_max / 2, row_max / 2
    min = None
    max = None
    # search min and max
    for row in range(len(image)):
        for col in range(len(image[0])):
            if np.sqrt((row - mid_x) ** 2 + (col - mid_y) ** 2) <= R:
                value = image[row][col]
                if min is None or min > value:
                    min = value

                if max is None or max < value:
                    max = value

    # fresh the new value
    for row in range(len(image)):
        for col in range(len(image[0])):
            if np.sqrt((row - mid_x) ** 2 + (col - mid_y) ** 2) <= R:
                new_image[row][col] = (image[row][col] - min) / (max - min) * 255

    return new_image


def train_test_split_with_labels(spectrum, label, numbers=[10, 10], random_state=0):
    data_list = [[], []]
    for spectrum_, label_ in zip(spectrum, label):
        data_list[label_].append(spectrum_)

    random.seed(random_state)
    idx_test_class0 = random.sample(range(0, len(data_list[0])), numbers[0])
    random.seed(random_state + 1)
    idx_test_class1 = random.sample(range(0, len(data_list[1])), numbers[1])
    for key, value in encoder_dict.items():
        if value == 0:
            print('{}:'.format(key), idx_test_class0)
        else:
            print('{}:'.format(key), idx_test_class1)

    X_test = [data_list[0][idx] for idx in idx_test_class0] + [data_list[1][idx] for idx in idx_test_class1]
    Y_test = [0] * len(idx_test_class0) + [1] * len(idx_test_class1)

    idx_train_class0 = [i for i in range(len(data_list[0]))]
    idx_train_class1 = [i for i in range(len(data_list[1]))]
    for idx in idx_test_class0:
        idx_train_class0.remove(idx)

    for idx in idx_test_class1:
        idx_train_class1.remove(idx)

    X_train = [data_list[0][idx] for idx in idx_train_class0] + [data_list[1][idx] for idx in idx_train_class1]
    Y_train = [0] * len(idx_train_class0) + [1] * len(idx_train_class1)

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test), data_list


def output_results(tar, pre, mark=False, acc_train=False, print_flag=False):
    matrix = confusion_matrix(tar, pre)
    Accuracy = accuracy_score(tar, pre)
    Precision = precision_score(tar, pre)
    Recall = recall_score(tar, pre)
    F1 = f1_score(tar, pre)
    Kappa = cohen_kappa_score(tar, pre)
    FP = matrix[0][1]
    TP = matrix[1][1]

    if TP == 0:
        TP = TP + 1e-8
    waste_ratio = FP / TP

    if print_flag:
        print("【Day %d】Train Accuracy: %.3f|Test accuracy: %.3f precsion: %.3f recall_score: %.3f f1_score: %.3f kappa: %.3f waste_ratio %.3f" % (mark, acc_train, Accuracy, Precision, Recall, F1, Kappa, waste_ratio))
        print('matrix: ', matrix)

    return Accuracy, Precision, Recall, F1, Kappa, waste_ratio, matrix


def output_results_simple(tar, pre):
    Accuracy = accuracy_score(tar, pre)
    Precision = precision_score(tar, pre)
    Recall = recall_score(tar, pre)

    return Accuracy, Precision, Recall


def plot_acc(scores, save_path):
    # plot results
    plt.figure(dpi=400)
    plt.clf()

    for label in ['train_accuracy', 'test_accuracy']:
        plt.plot(scores[label], label=label)

    plt.legend()
    plt.savefig(save_path)


def plot_loss(scores, save_path):
    plt.figure(dpi=400)
    plt.clf()

    for label in ['train_loss', 'test_loss']:
        plt.plot(scores[label], label=label)

    plt.legend()
    plt.savefig(save_path)


def check_dataset(input_dir):
    first_name = os.listdir(input_dir)[0]
    first_path = os.path.join(input_dir, first_name)

    data = np.load(first_path)
    print('data shape:', data.shape)

    return data.shape[2]


class scores_save_and_average:
    def __init__(self):
        self.score_names = ['accuracy', 'precision', 'recall', 'f1', 'kappa', 'waste_ratio', 'matrix']
        self.score_dict = {
            score_name: []
            for score_name in self.score_names
        }
        self.score_mean_dict = {
            score_name: 0
            for score_name in self.score_names
        }

    def get_mean(self, print_flag=False):
        for score_name, score_value in self.score_dict.items():
            if score_name == 'matrix':
                self.score_mean_dict[score_name] = np.array(self.score_dict[score_name])
                continue
            self.score_mean_dict[score_name] = np.mean(np.array(self.score_dict[score_name]))

        if print_flag:
            print('\n【mean score】 ')
            for score_name, score_value in self.score_mean_dict.items():
                if score_name != 'matrix':
                    score_value = round(score_value, 3)
                print(score_name, score_value)

        return self.score_mean_dict.values()

    def fresh_value(self, scores):
        for score_name, score_ in zip(self.score_names, scores):
            self.score_dict[score_name].append(score_)

    def print_max(self):
        max_value, max_index = self.find_max_value_and_index(self.score_dict['accuracy'], self.score_dict['precision'])

        print('max_acc_idx ', max_index,
              'max_acc ', round(max_value, 3), 'max_prec ', round(self.score_dict['precision'][max_index], 3))

        return max_index

    def find_max_value_and_index(self, a, b):
        max_value = float('-inf')  # 初始化最大值为负无穷
        max_index = -1

        for i in range(len(a)):
            if a[i] > max_value or (a[i] == max_value and b[i] > b[max_index]):
                max_value = a[i]
                max_index = i

        return max_value, max_index

    def return_cv_idx_scores(self, cv_idx):
        print('\n【max score】 ')
        for score_name, score_value in self.score_dict.items():
            if score_name != 'matrix':
                score_value = round(score_value[cv_idx], 3)

            print(score_name, score_value)


def check_system():
    flag_apply_for_device = False
    system_name = platform.system()
    if system_name == "Windows":
        flag_apply_for_device = True
    elif system_name == "Linux":
        flag_apply_for_device = False
    else:
        print("当前系统是 %s" % system_name)
    if flag_apply_for_device:
        input_root_dir = 'D:/spector_former_in_ovo_sexing/egg_data_npy/8d/'  # 'D:\\spector_former_in_ovo_sexing\\egg_data_npy'
    else:
        input_root_dir = '/nfs/my/Xu/jicm/food_datasets_npy/10d/'

    return input_root_dir


def check_system_return_bool():
    system_name = platform.system()
    if system_name == "Windows":
        return True
    elif system_name == "Linux":
        return False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameters_millions(model):
    num_params = count_parameters(model)
    print(f"Number of training parameters: {num_params / 1e6:.2f} M")


def check_dirs(dirs_list):
    for dir_ in dirs_list:
        check_dir(dir_)


def return_wavelength_idx(input_wavelengths):
    wavelength_idx = None
    if input_wavelengths == 'all':
        wavelength_idx = False
    if input_wavelengths == 'rf':
        wavelength_idx = return_bands_RF()
    if input_wavelengths == 'spa':
        wavelength_idx = return_bands_SPA()
    if input_wavelengths == 'cars':
        wavelength_idx = return_bands_CARS()
    if input_wavelengths == 'pca_channel':
        wavelength_idx = return_bands_PCA_channel()
    if input_wavelengths == 'pca_img':
        wavelength_idx = 'pca_img'

    return wavelength_idx


