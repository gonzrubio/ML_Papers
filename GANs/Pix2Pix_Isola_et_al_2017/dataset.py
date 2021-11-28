"""Image-to-Image Translation with Conditional Adversarial Networks.

Dataset: https://www.kaggle.com/defileroff/comic-faces-paired-synthetic

Created on Mon Nov 22 17:15:42 2021

@author: gonzr
"""


import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def mean_std(dataset):
    """Return the mean and std of the dataset."""
    loader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)

    mean_inputs = 0.
    std_inputs = 0.
    mean_targets = 0.
    std_targets = 0.

    for inputs, targets in tqdm(loader):

        inputs = inputs.to(DEVICE).view(inputs.size(0), inputs.size(1), -1)
        mean_inputs += inputs.mean(2).sum(0)
        std_inputs += inputs.std(2).sum(0)

        targets = targets.to(DEVICE).view(targets.size(0), inputs.size(1), -1)
        mean_targets += targets.mean(2).sum(0)
        std_targets += targets.std(2).sum(0)

    mean_inputs /= len(loader.dataset)
    std_inputs /= len(loader.dataset)
    mean_targets /= len(loader.dataset)
    std_targets /= len(loader.dataset)

    return (mean_inputs, std_inputs), (mean_targets, std_targets)


class Face2Comic(Dataset):
    """A paired face-to-comics dataset."""

    def __init__(self, data_dir, train=True):
        super(Face2Comic, self).__init__()
        self.data_dir = data_dir
        self.faces_dir = os.path.join(data_dir, "faces")
        self.faces = os.listdir(self.faces_dir)
        self.comics_dir = os.path.join(data_dir, "comics")
        self.comics = os.listdir(self.comics_dir)
        self.len = len(self.faces)
        self.train = train

    def apply_transforms(self, face, comic):
        """Apply the same transforms to the input and the target."""
        common_transform = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor()])

        normalize_face = transforms.Normalize(mean=[0.5129, 0.4136, 0.3671],
                                              std=[0.2372, 0.1972, 0.1883])

        normalize_comic = transforms.Normalize(mean=[0.4445, 0.3650, 0.3226],
                                               std=[0.2594, 0.2051, 0.1840])

        face = normalize_face(common_transform(face))
        comic = normalize_comic(common_transform(comic))

        if self.train:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter()])
            face = train_transform(face)
            comic = train_transform(comic)

        return face, comic

    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.len

    def __getitem__(self, index):
        """Return the transformed input (face) and target (comic)."""
        face = Image.open(os.path.join(self.faces_dir, self.faces[index]))
        comic = Image.open(os.path.join(self.comics_dir, self.comics[index]))
        return self.apply_transforms(face, comic)


if __name__ == '__main__':

    data_dir_train = os.getcwd() + '\\data\\train\\'
    dataset_train = Face2Comic(data_dir=data_dir_train, train=True)

    stats_faces, stats_comics = mean_std(dataset_train)

    print(f"Faces: mean = {stats_faces[0]}, std = {stats_faces[1]}")
    print(f"Comics: mean = {stats_comics[0]}, std = {stats_comics[1]}")

    data_dir_val = os.getcwd() + '\\data\\val\\'
    dataset_val = Face2Comic(data_dir=data_dir_val, train=False)

    stats_faces, stats_comics = mean_std(dataset_val)

    print(f"Faces: mean = {stats_faces[0]}, std = {stats_faces[1]}")
    print(f"Comics: mean = {stats_comics[0]}, std = {stats_comics[1]}")
