import os
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class SARDataLoader(Dataset):
    def __init__(self, root, data, train=True):
        super(SARDataLoader).__init__()
        self.totensor = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])  # 转为灰度图像
        self.root = os.path.expanduser(root)
        self.train = train
        self.image_data = []
        self.mask_data = []
        self.sat_data = []
        self.gt_data = []
        if self.train:
            file_names = [data]
            for file_name in file_names:
                train_path = os.path.join(root, file_name)
                image_path = os.path.join(train_path, 'image')
                mask_path = os.path.join(train_path, 'mask')
                for f in os.listdir(image_path):
                    f = os.path.join(image_path, f)
                    self.image_data.append(f)
                for f in os.listdir(mask_path):
                    f = os.path.join(mask_path, f)
                    self.mask_data.append(f)
        else:
            file_names = [data]
            for file_name in file_names:
                test_path = os.path.join(root, file_name)
                sat_path = os.path.join(test_path, 'sat')
                gt_path = os.path.join(test_path, 'gt')
                for f in os.listdir(sat_path):
                    f = os.path.join(sat_path, f)
                    self.sat_data.append(f)
                for f in os.listdir(gt_path):
                    f = os.path.join(gt_path, f)
                    self.gt_data.append(f)

    def __getitem__(self, idx):
        if self.train:
            out_image_addr = self.image_data[idx]
            out_image_arr = cv2.imread(out_image_addr)

            out_mask_addr = self.mask_data[idx]
            out_mask_arr = cv2.imread(out_mask_addr)

            return self.totensor(out_image_arr), self.totensor(out_mask_arr)
        else:
            out_sat_addr = self.sat_data[idx]
            out_sat_arr = cv2.imread(out_sat_addr)

            out_gt_addr = self.gt_data[idx]
            out_gt_arr = cv2.imread(out_gt_addr)

            return self.totensor(out_sat_arr), self.totensor(out_gt_arr)

    def __len__(self):
        if self.train:
            length = len(self.image_data)
        else:
            length = len(self.sat_data)
        return length
