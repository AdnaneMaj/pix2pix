import numpy as np
import pix2pix.config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir_input = "/kaggle/input/driveeee/Drive_data/seg/seg",root_dir_target = "/kaggle/input/driveeee/Drive_data/rgb/images"):
        self.root_dir_input = root_dir_input
        self.root_dir_target = root_dir_target
        self.list_files_input = os.listdir(self.root_dir_input)
        self.list_files_target = os.listdir(self.root_dir_target)
        

    def __len__(self):
        return max(len(self.list_files_input),500)

    def __getitem__(self, index):
        index = index % len(self.list_files_input)
        img_file_input,img_file_target = self.list_files_input[index],self.list_files_target[index]
        img_path_input = os.path.join(self.root_dir_input, img_file_input)
        img_path_target = os.path.join(self.root_dir_target, img_file_target)
        input_image = np.array(Image.open(img_path_input))
        target_image = np.array(Image.open(img_path_target))

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset()
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
