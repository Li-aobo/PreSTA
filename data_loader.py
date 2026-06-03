import torch
import torchvision
import folders


class DataLoader(object):

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, Info=None):

        self.batch_size = batch_size
        self.istrain = istrain

        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=patch_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

        if dataset == 'kadid-10k':
            if istrain:
                self.data = folders.KADID_10kCorSelFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num, Info=Info)
            else:
                self.data = folders.KADID_10kFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            if istrain:
                self.data = folders.Koniq_10kCorSelFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num, Info=Info)
            else:
                self.data = folders.Koniq_10kFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num)

        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'bid':
            self.data = folders.BIDFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'spaq':
            if istrain:
                self.data = folders.SPAQCorSelFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num, Info=Info)
            else:
                self.data = folders.SPAQFolder(
                    root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)


    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size * 4, shuffle=False)
        return dataloader


class FeatureDataLoader(object):

    def __init__(self, dataset, path, img_indx, patch_size, patch_num=1,
                 batch_size=32, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        self.data = build_feature_dataset(dataset, path, img_indx, transforms, patch_num)

    def get_data(self):
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def build_feature_dataset(dataset, path, img_indx, transforms, patch_num):
    if dataset == 'kadid-10k':
        return folders.KADID_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'koniq-10k':
        return folders.Koniq_10kFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'livec':
        return folders.LIVEChallengeFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'bid':
        return folders.BIDFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'spaq':
        return folders.SPAQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'live':
        return folders.LIVEFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'csiq':
        return folders.CSIQFolder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    if dataset == 'tid2013':
        return folders.TID2013Folder(
            root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    raise ValueError('Unsupported dataset for feature extraction: {}'.format(dataset))
