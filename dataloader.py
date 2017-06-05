
import torch
from torchvision import transforms
import torch.utils.data as data

from PIL import Image
import os
import os.path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    #counter = 0
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
                #counter += 1
                #if counter > 100:
                #    return images

    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        print("Found {} images in subfolders of: {}".format(len(imgs), root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

def get_loader(root, split, batch_size, scale_size, num_workers=12, shuffle=True):
    dataset_name = os.path.basename(root)
    image_root = os.path.join(root, 'splits', split)

    if dataset_name in ['CelebA']:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    else:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader


