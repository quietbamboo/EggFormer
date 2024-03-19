import argparse, os, sys
sys.path.insert(0, sys.path[0] + "/../")
current_dir = os.path.dirname(sys.argv[0])
from sklearn.model_selection import StratifiedKFold
from utils import *
from run_eggformer import fine_tune_vit
from gard_model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# seed
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0), self.h, self.w, x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


if __name__ == '__main__':
    # setting
    args = argparse.ArgumentParser()
    args.add_argument('--mode', default='eggformer', help='eggformer or vit')
    args.add_argument('--num_classes', default=2)
    args.add_argument('--cv_nums', default=3)
    args.add_argument('--weights', default=current_dir + '/vit_base_patch16_224_in21k.pth')
    args.add_argument('--eggFormer-weights', default=current_dir+'/saved_models/seed0_cv2_ep79_acc0.953_prec0.957.pth')

    args.add_argument('--freeze-layers', default=True)
    opt = args.parse_args()
    check_dir(current_dir+'/grad_cam/')

    # create model
    model = fine_tune_vit(opt, in_c=440)  # test

    # load dict
    model_dict = torch.load(opt.eggFormer_weights)
    model.load_state_dict(model_dict)
    model.eval()

    # load dataset
    input_dir = check_system()
    input_file_paths, input_file_labels = return_path_labels(input_dir)
    input_file_paths, input_file_labels = np.array(input_file_paths), np.array(input_file_labels)

    skf_val = StratifiedKFold(n_splits=opt.cv_nums, shuffle=True, random_state=0)
    for cv_idx, (train_idx, test_idx) in enumerate(skf_val.split(input_file_paths, input_file_labels)):
        if cv_idx < 2:
            continue

        train_x, train_y, test_x_paths, test_y = input_file_paths[train_idx], input_file_labels[train_idx], input_file_paths[test_idx], input_file_labels[test_idx]

        pd.to_pickle(test_x_paths, current_dir + '/test_x_paths.pkl')
        test_x = []
        for test_x_path in test_x_paths:
            data = np.load(test_x_path)
            data = cv2.resize(data, dsize=(224, 224))
            test_x.append(resize_imgs(data))

        test_x = np.array(test_x).astype(np.float32)

        # test
        pred = model(torch.FloatTensor(test_x).to(device))
        pred_class = torch.max(pred, dim=1)[1]
        label = torch.LongTensor(test_y).to(device)
        acc = accuracy_score(test_y, pred_class.detach().cpu().numpy())
        print(f'\nacc: {acc:.3f}')

        # grad cam
        target_layers = [model.blocks[-1].norm1]
        cam = GradCAM(model=model,
                      target_layers=target_layers,
                      use_cuda=True,
                      reshape_transform=ReshapeTransform(model))

        target = 0
        for idx, (input_np, label) in enumerate(zip(test_x, test_y)):
            if pred_class[idx] == 1-target:
                continue

            input_tensor = torch.FloatTensor(input_np).to(device)
            input_tensor = input_tensor.unsqueeze(0)
            
            grayscale_cam = cam(input_tensor=input_tensor, target_category=int(label))

            grayscale_cam = grayscale_cam[0, :]

            img_rgb = cv2.merge([input_np[226, :, :], input_np[108, :, :], input_np[50, :, :]])
            img_rgb = img_rgb / np.max(img_rgb)

            visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
            cam_save_path = current_dir+'/grad_cam/'+os.path.basename(test_x_paths[idx]).split('.')[0] + 'pred_{}.png'.format(pred_class[idx])
            rgb_save_path = current_dir+'/grad_cam/'+os.path.basename(test_x_paths[idx]).split('.')[0] + 'pred_{}_ori.png'.format(pred_class[idx])

            cv2.imwrite(rgb_save_path, img_rgb * 255)
            cv2.imwrite(cam_save_path, visualization)


            # plt.title(os.path.basename(test_x_paths[idx]) + ' pred: {}'.format(target))
            # plt.axis('off')   # 去坐标轴
            # plt.xticks([])    # 去 x 轴刻度
            # plt.yticks([])    # 去 y 轴刻度
            # plt.imshow(visualization)
            # plt.savefig(cam_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            #
            # plt.axis('off')   # 去坐标轴
            # plt.xticks([])    # 去 x 轴刻度
            # plt.yticks([])    # 去 y 轴刻度
            # plt.imshow(img_rgb)
            # plt.savefig(rgb_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            # plt.show()




