import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

modelnet10_label = np.array([2, 3, 9, 13, 15, 23, 24, 31, 34, 36]) - 1

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]

def _load_data_file(name, subset10):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:].astype(int)
    if subset10:
        label_list = modelnet10_label.tolist()
        valid_list = []
        for i in range(len(label)):
            if label[i] in label_list:
                valid_list.append(i)
                idx = label_list.index(label[i])
                label[i] = idx
        valid_list = np.array(valid_list)
        data = data[valid_list]
        label = label[valid_list]
    return data, label


class TerrcottaCls(data.Dataset):

    def __init__(
            self, transforms=None, train=True, self_supervision=False, subset10=False, use_normal=False, dataset_rate=1.0,reconstruct=False,scale=0
    ):
        super().__init__()

        self.transforms = transforms

        self.self_supervision = self_supervision

        self.train = train

        self.reconstruct = reconstruct

        root = './data/Terracotta_warriors_normal/'
        # root = './dataset/modelnet40_normal_resampled/'
        if reconstruct:
        # ——重建
            root = './Terracotta_warriors_DATA-1123/'
        if subset10:
            if self_supervision:
                self.points = np.load(root + 'terr_normal_2048_train_points.npy')
                self.labels = None
            elif train:
                self.points = np.load(root + 'terr_normal_2048_train_points.npy')
                self.labels = np.load(root + 'terr_normal_2048_train_label.npy')
            else:
                self.points = np.load(root + 'terr_normal_2048_test_points.npy')
                self.labels = np.load(root + 'terr_normal_2048_test_label.npy')
        else:
            if self_supervision:
                self.points = np.load(root + 'terr_normal_2048_train_points.npy')
                self.labels = None
            elif train:
                if scale>0:
                    # self.points = np.load(root + 'terr_scale_2048_train_points.npy')
                    # self.labels = np.load(root + 'terr_scale_2048_train_label.npy')
                    self.points = np.load(root + 'terr_guass' + str(scale) + '_2048_train_points_o.npy')
                    self.labels = np.load(root + 'terr_guass' + str(scale) + '_2048_train_label_o.npy')
                else:
                    self.points = np.load(root + 'trainpoint.npy')
                    self.labels = np.load(root + 'trainlabel.npy')
                    # self.points = np.load(root + 'terr_guass0.04_2048_train_points.npy')
                    # self.labels = np.load(root + 'terr_guass0.04_2048_train_label.npy')
            elif reconstruct:
                # self.points = np.load(root + 'terr_normal_2048_test_points.npy')
                # self.labels = np.load(root + 'terr_normal_2048_test_label.npy')
                # # ——重建
                # self.save_paths = np.load(root + 'terr_normal_2048_test_savepath.npy')
                # self.save_sampled_paths = np.load(root + 'terr_normal_2048_test_save_sample1024_path.npy')

                self.points = np.load(root + 'recon_normal_2048_test_points.npy')
                self.labels = np.load(root + 'recon_normal_2048_test_label.npy')
                # ——重建
                self.save_paths = np.load(root + 'recon_normal_2048_test_savepath.npy')
                self.save_sampled_paths = np.load(root + 'recon_normal_2048_test_save_sample1024_path.npy')
            else:
                if scale>0:
                    # self.points = np.load(root + 'terr_scale_2048_test_points.npy')
                    # self.labels = np.load(root + 'terr_scale_2048_test_label.npy')
                    self.points = np.load(root + 'terr_guass'+str(scale)+'_2048_test_points.npy')
                    self.labels = np.load(root + 'terr_guass'+str(scale)+'_2048_test_label.npy')

                else:
                    self.points = np.load(root + 'testpoint.npy')
                    self.labels = np.load(root + 'testlabel.npy')
                    print('**********************')


        if not use_normal:
            self.points = self.points[:, :, :3]

        if dataset_rate < 1.0:
            print('### ATTENTION: Only', dataset_rate, 'data of training set are used ###')
            num_instance = len(self.labels)
            index = np.random.permutation(num_instance)[0: int(num_instance * dataset_rate)]
            self.points = self.points[index]
            if self.labels is not None:
                self.labels = self.labels[index]

        if not subset10:
            print('Successfully load ModelNet40 with', self.points.shape[0], 'instances')
        else:
            print('Successfully load ModelNet10 with', self.points.shape[0], 'instances')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        if self.self_supervision:
            return current_points
        elif self.reconstruct:
            label = self.labels[idx]
        # ——重建
            save_path = self.save_paths[idx]
            save_sampled_path = self.save_sampled_paths[idx]
            return current_points, label, save_path, save_sampled_path
        else:
            label = self.labels[idx]
            return current_points, label


    def __len__(self):
        return self.points.shape[0]

