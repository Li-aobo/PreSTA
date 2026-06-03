import os

import numpy as np
import pandas as pd
import scipy.io
import torch.utils.data as data
from PIL import Image


class IQAFolder(data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path, target, mos, std = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target, mos, std

    def __len__(self):
        return len(self.samples)


class LIVEFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        refname = get_file_names(os.path.join(root, 'refimgs'), '.bmp')

        imgpath = []
        imgpath += get_live_distortion_file_names(os.path.join(root, 'jp2k'), 227)
        imgpath += get_live_distortion_file_names(os.path.join(root, 'jpeg'), 233)
        imgpath += get_live_distortion_file_names(os.path.join(root, 'wn'), 174)
        imgpath += get_live_distortion_file_names(os.path.join(root, 'gblur'), 174)
        imgpath += get_live_distortion_file_names(os.path.join(root, 'fastfading'), 174)

        info = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        dmos = info['dmos_new'].astype(np.float32)[0]
        std = info['dmos_std'].astype(np.float32)[0]
        labels = normalize_labels(dmos, flip=True)

        orgs = info['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))['refnames_all']

        samples = []
        for ref_idx in index:
            selected = (refname[ref_idx] == refnames_all) & ~orgs.astype(np.bool_)
            selected = np.where(selected)[1].tolist()
            for item in selected:
                for _ in range(patch_num):
                    samples.append((imgpath[item], labels[item], -dmos[item], std[item]))
        super().__init__(samples, transform)


class LIVEChallengeFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))['AllImages_release'][7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))[
            'AllMOS_release'].astype(np.float32)[0][7:1169]
        std = scipy.io.loadmat(os.path.join(root, 'Data', 'AllStdDev_release.mat'))[
            'AllStdDev_release'].astype(np.float32)[0][7:1169]
        labels = normalize_labels(mos)

        samples = []
        for item in index:
            image_name = imgpath[item][0][0]
            for _ in range(patch_num):
                samples.append((os.path.join(root, 'Images', image_name), labels[item], mos[item], std[item]))
        super().__init__(samples, transform)


class CSIQFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        data_info = scipy.io.loadmat(os.path.join(root, 'CSIQ_info.mat'))
        std_info = scipy.io.loadmat(os.path.join(root, 'csiq_std_mos.mat'))

        dst_name_array = data_info['dst_name'][:, 0]
        dmos = data_info['mos'][:, 0].astype(np.float32)
        std = std_info['std_mos'][:, 0].astype(np.float32)
        labels = normalize_labels(dmos, flip=True)

        refname = get_file_names(os.path.join(root, 'src_imgs'), '.png')

        imgnames = []
        refnames_all = []
        for item in dst_name_array:
            dst_name = item[0]
            imgnames.append(dst_name)
            ref_temp = dst_name.split('/')[-1].split('.')
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])
        refnames_all = np.array(refnames_all)

        samples = []
        for ref_idx in index:
            selected = np.where(refname[ref_idx] == refnames_all)[0].tolist()
            for item in selected:
                for _ in range(patch_num):
                    samples.append((os.path.join(root, imgnames[item]), labels[item], -dmos[item], std[item]))
        super().__init__(samples, transform)


class Koniq_10kFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        info = pd.read_csv(os.path.join(root, 'koniq10k_scores_and_distributions.csv'))
        imgname = info['image_name'].tolist()
        mos = info['MOS'].values.astype(np.float32)
        std = info['SD'].values.astype(np.float32)
        labels = normalize_labels(mos)

        samples = []
        for item in index:
            for _ in range(patch_num):
                samples.append((os.path.join(root, '512x384', imgname[item]), labels[item], mos[item], std[item]))
        super().__init__(samples, transform)


class BIDFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        info = pd.read_excel(os.path.join(root, 'DatabaseGrades.xlsx'))
        img_num = info['Image Number'].tolist()
        imgname = ['DatabaseImage%04d.JPG' % item for item in img_num]
        mos = info['Average Subjective Grade'].values.astype(np.float32)
        std = info['std'].values.astype(np.float32)
        labels = normalize_labels(mos)

        samples = []
        for item in index:
            for _ in range(patch_num):
                samples.append((os.path.join(root, imgname[item]), labels[item], mos[item], std[item]))
        super().__init__(samples, transform)


class TID2013Folder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        refname = get_tid_file_names(os.path.join(root, 'reference_images'), '.bmp.BMP')

        imgnames = []
        target = []
        refnames_all = []
        with open(os.path.join(root, 'mos_with_names.txt'), 'r') as f:
            for line in f.readlines():
                words = line.strip().split()
                target.append(words[0])
                imgnames.append(words[1])
                refnames_all.append(words[1].split('_')[0][1:])

        mos = np.array(target).astype(np.float32)
        with open(os.path.join(root, 'mos_std.txt'), 'r') as f:
            std = np.array(f.readlines()).astype(np.float32)
        labels = normalize_labels(mos)
        refnames_all = np.array(refnames_all)

        samples = []
        for ref_idx in index:
            selected = np.where(refname[ref_idx] == refnames_all)[0].tolist()
            for item in selected:
                for _ in range(patch_num):
                    samples.append(
                        (os.path.join(root, 'distorted_images', imgnames[item]), labels[item], mos[item], std[item]))
        super().__init__(samples, transform)


class KADID_10kFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        refname = ['I%02d.png' % item for item in range(1, 82)]

        info = pd.read_csv(os.path.join(root, 'dmos.csv'))
        imgnames = info['dist_img'].tolist()
        refnames_all = info['ref_img'].values
        mos = info['dmos'].values.astype(np.float32)
        std = np.sqrt(info['var'].values.astype(np.float32))
        labels = normalize_labels(mos)

        samples = []
        for ref_idx in index:
            selected = np.where(refname[ref_idx] == refnames_all)[0].tolist()
            for item in selected:
                for _ in range(patch_num):
                    samples.append((os.path.join(root, 'images', imgnames[item]), labels[item], mos[item], std[item]))
        super().__init__(samples, transform)


class Koniq_10kCorSelFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num, Info):
        samples = build_corsel_samples(root, '512x384', Info, patch_num)
        super().__init__(samples, transform)


class SPAQCorSelFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num, Info):
        samples = build_corsel_samples(root, 'TestImage', Info, patch_num)
        super().__init__(samples, transform)


class KADID_10kCorSelFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num, Info):
        samples = build_corsel_samples(root, 'images', Info, patch_num)
        super().__init__(samples, transform)


class SPAQFolder(IQAFolder):
    def __init__(self, root, index, transform, patch_num):
        info = pd.read_excel(os.path.join(root, 'Annotations', 'MOS and Image attribute scores.xlsx'))
        imgname = info['Image name'].tolist()
        mos = info['MOS'].values.astype(np.float32)
        labels = normalize_labels(mos)

        samples = []
        for item in index:
            for _ in range(patch_num):
                samples.append((os.path.join(root, 'TestImage', imgname[item]), labels[item], mos[item], 0.0))
        super().__init__(samples, transform)


def build_corsel_samples(root, image_dir, info, patch_num):
    if info is None:
        raise ValueError('CorSel training requires selected sample information.')

    imgnames = info['names']
    labels = np.array(info['gts'], dtype=np.float32)

    samples = []
    for item, image_name in enumerate(imgnames):
        for _ in range(patch_num):
            samples.append((os.path.join(root, image_dir, image_name), labels[item], 0.0, 0.0))
    return samples


def get_live_distortion_file_names(path, num):
    return [os.path.join(path, 'img%s.bmp' % (idx + 1)) for idx in range(num)]


def get_file_names(path, suffix):
    return [item for item in sorted(os.listdir(path)) if os.path.splitext(item)[1] == suffix]


def get_tid_file_names(path, suffix):
    return [item[1:3] for item in sorted(os.listdir(path)) if suffix.find(os.path.splitext(item)[1]) != -1]


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def normalize_labels(ys, flip=False):
    assert type(ys) == np.ndarray
    y_max = np.max(ys)
    y_min = np.min(ys)
    ys_norm = (ys - y_min) / (y_max - y_min)
    if flip:
        ys_norm = 1 - ys_norm
    return ys_norm * 10.0
