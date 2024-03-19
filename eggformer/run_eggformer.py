import sys, os
sys.path.insert(0, sys.path[0] + "/../")
current_dir = os.path.dirname(sys.argv[0])
import math
from tqdm import trange
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse
from utils import *
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold
from eggformer.vit_model import vit_base_patch16_224_in21k as create_vit_model


def fine_tune_vit(opt, in_c, device='cuda:0'):
    weights = opt.weights
    freeze_layers = opt.freeze_layers

    model = create_vit_model(num_classes=opt.num_classes, has_logits=False, in_c=in_c, mode=opt.mode).to(device)
    if weights != "":
        weights_dict = torch.load(weights, map_location=device)
        # 删除不需要的权重
        for k in ['head.weight', 'head.bias', 'pre_logits.fc.bias', 'patch_embed.proj.weight', 'patch_embed.proj.bias',
                  'pre_logits.fc.weight']:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if freeze_layers:
        require_grads_layers = ["head", "pre_logits", "pos_embed", "SE_module", "PD_conv", "patch_embed", "cls_token"]  # , "blocks.10", "blocks.11",
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结 'cls_token' 'pos_embed' 'patch_embed.proj.weight' 'patch_embed.proj.bias'
            if not check_name(name, require_grads_layers):  # todo
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    return model


def check_name(name, namelist):
    exist_flag = False
    for name_ in namelist:
        if name_ in name:
            exist_flag = True
            break
    return exist_flag


def create_model(opt, in_c):
    model = fine_tune_vit(opt=opt, in_c=in_c).to(device)
    print_parameters_millions(model)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=opt.lr, weight_decay=5E-5)  # optimizer
    # optimizer = optim.SGD(pg, lr=opt.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_transform = DataAugmentation().to(device)

    return model, scheduler, optimizer, train_transform


def train_one_epoch(model, optimizer, data_loader):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data[0].to(device), data[1].to(device)
        sample_num += images.shape[0]
        if opt.img_augment:
            images = train_transform(images)

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        # 在更新权重之前，对梯度进行裁剪，使其不超过0.5
        torch.nn.utils.clip_grad_value_([p for p in model.parameters() if p.requires_grad], clip_value=0.8)
        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def evaluate(model, data_loader):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_TP, accu_P = torch.zeros(1).to(device), torch.zeros(1).to(device)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data[0].to(device), data[1].to(device)
        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels).sum()
        accu_TP += torch.sum((pred_classes == 1) & (labels == 1))
        accu_P += torch.sum(pred_classes == 1)

        loss = loss_function(pred, labels)
        accu_loss += loss

    accu_P_num = accu_P.item()
    if accu_P_num == 0:
        accu_P_num = 1e-6
    prec = accu_TP.item() / accu_P_num

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, prec


if __name__ == '__main__':
    # args
    args = argparse.ArgumentParser()
    args.add_argument('--data_size', default=224, help='data size')
    args.add_argument('--num_classes', default=2)

    # channel attention
    args.add_argument('--channel_ratio', default=4)
    args.add_argument('--batch_size', default=64, help='batch size')
    args.add_argument('--epochs', default=120, help='epochs')
    args.add_argument('--img_augment', default=True)
    args.add_argument('--cv_nums', default=3)
    args.add_argument('--lr', default=6e-4)
    args.add_argument('--lrf', default=0.01)

    args.add_argument('--input-days', default=[10])  # [0, 2, 4, 6, 8, 10, 12, 14]
    args.add_argument('--input-wavelengths', default='all', help='all rf pca_img pca_channel spa cars')
    args.add_argument('--weights', default=current_dir + '/vit_base_patch16_224_in21k.pth')
    args.add_argument('--freeze-layers', default=True)
    args.add_argument('--mode', default='eggformer', help='eggformer or vit')
    args.add_argument('--seed', default=0)
    args.add_argument('--device', default='cuda:0')
    opt = args.parse_args()

    # # Parameter Setting
    seed = opt.seed
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # main
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    input_root_dir = check_system()  # todo modify

    # test create model
    model = create_model(opt, 440)[0]
    summary(model, (440, 224, 224))

    days = opt.input_days
    flag_all_days = True if len(days) > 1 else False

    for day in days:
        if check_system_return_bool():
            day = 8
        input_dir = input_root_dir + '/../{}d/'.format(day)
        save_dir = current_dir + '/{}/{}d/'.format(opt.mode, day)
        check_dirs([save_dir + '/models/', save_dir + '/plot/'])

        out = pd.read_pickle(current_dir+'/saved_models/out.pkl')
        out = out.reshape(out.shape[0], 440)
        out_mean = np.mean(out, axis=0)
        out_mean_sort = np.argsort(out_mean*(-1))
        out_mean_norm = out_mean/np.sum(out_mean)

        selected_num = 2
        used_flag = [False] * selected_num

        idx_list = []
        account = 0
        for idx in out_mean_sort:
            if account + out_mean_norm[idx] < 0.95 and out_mean_norm[idx] > 3e-3:
                if used_flag[idx // int(440/selected_num)] == True:
                    continue
                else:
                    used_flag[idx // int(440/selected_num)] = True

                idx_list.append(idx)
                account += out_mean_norm[idx]
            else:
                break
        print('len list: ', len(idx_list))

        # datasets
        dataset_x, dataset_y = return_dataset_with_wavelength_idx(input_dir, dsize=(opt.data_size, opt.data_size), wavelength_idx=idx_list)  # wavelength_idx=return_wavelength_idx(opt.input_wavelengths)

        skf_val = StratifiedKFold(n_splits=opt.cv_nums, shuffle=True, random_state=0)
        fold_scores = scores_save_and_average()
        log_score_names = ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']

        # cross validation
        for cv_idx, (train_idx, test_idx) in enumerate(skf_val.split(dataset_x, dataset_y)):
            # if cv_idx < 2:
            #     continue
            log_score_data = {score_name: [] for score_name in log_score_names}
            # split dataset
            train_x, train_y, test_x, test_y = dataset_x[train_idx], dataset_y[train_idx], dataset_x[test_idx], dataset_y[test_idx]

            # 实例化训练数据集
            train_loader = return_data_loader(train_x, train_y, batch_size=opt.batch_size, shuffle=True, seed=seed)  # data loader
            test_loader = return_data_loader(test_x, test_y, batch_size=opt.batch_size, shuffle=False, seed=seed)  # data loader

            # 初始化model 优化器等
            model, scheduler, optimizer, train_transform = create_model(opt, in_c=train_x[0].shape[0])
            model = model.to(device)

            # 初始化 early_stopping 对象
            best_acc, best_prec, best_scores = 0, 0, []
            previous_model_save_path = ''
            for epoch_idx in trange(opt.epochs):
                # train
                tr_loss, tr_acc = train_one_epoch(model, optimizer, train_loader)
                scheduler.step()

                # evaluate
                te_loss, te_acc, te_prec = evaluate(model, test_loader)
                print(f"\n[{day:d}d_CV{cv_idx:d} Seed{opt.seed:d} Ep{epoch_idx:d}] train_loss {tr_loss:.3f} test_loss {te_loss:.3f} train_acc {tr_acc:.3f} test_acc {te_acc:.3f} test_prec {te_prec:.3f}")

                # save log && plot
                for score_name, score_value in zip(log_score_names, [tr_loss, te_loss, tr_acc, te_acc]):
                    log_score_data[score_name].append(score_value)

                if te_acc > best_acc or (te_acc == best_acc and te_prec >= best_prec):
                    best_acc, best_prec = te_acc, te_prec

                    te_pred = model(torch.FloatTensor(test_x).to(device))
                    te_pred_class = torch.max(te_pred, dim=1)[1].cpu().numpy()

                    te_scores = output_results(test_y, te_pred_class, mark=10, acc_train=tr_acc, print_flag=False)
                    best_scores = te_scores
                    assert round(best_scores[0], 3) == round(te_acc, 3), f"test acc {te_acc:.3f}不一致 outputs {best_scores[0]:.3f}"

                    if not flag_all_days:
                        model_save_path = save_dir + f'/models/seed{opt.seed:d}_cv{cv_idx:d}_ep{epoch_idx:d}_acc{te_acc:.3f}_prec{te_prec:.3f}.pth'
                        torch.save(model.state_dict(), model_save_path)
                        if previous_model_save_path != '':
                            os.remove(previous_model_save_path)
                            previous_model_save_path = model_save_path

            # epochs finished
            fold_scores.fresh_value(best_scores)
            plot_acc(log_score_data, save_dir + f'/plot/seed{opt.seed:d}_cv{cv_idx:d}_acc.jpg')
            plot_loss(log_score_data, save_dir + f'/plot/seed{opt.seed:d}_cv{cv_idx:d}_loss.jpg')

            pd.to_pickle(log_score_data, save_dir + f'/seed{opt.seed:d}_cv{cv_idx:d}_scores.pkl')

        # TODO: cvs finished
        mean_values = fold_scores.get_mean(print_flag=True)
        cv_acc_max_idx = fold_scores.print_max()
        print('best scores in cvs: ')
        fold_scores.return_cv_idx_scores(cv_acc_max_idx)
        pd.DataFrame(fold_scores.score_dict).to_excel(save_dir + '/fold_scores.xlsx')


