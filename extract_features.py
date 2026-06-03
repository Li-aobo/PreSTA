import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from baseline import HierarchicalBackbone
from base_tools import dump_pkl, get_db_base_info, makedirs, seed_torch
from data_loader import FeatureDataLoader


def to_list(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def extract_dataset_features(config, dataset):
    folder_path, sel_num = get_db_base_info(config.root, dataset)
    img_data = FeatureDataLoader(
        dataset,
        folder_path,
        sel_num,
        config.patch_size,
        patch_num=1,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    ).get_data()

    device = torch.device('cuda' if torch.cuda.is_available() and config.device >= 0 else 'cpu')
    model = HierarchicalBackbone(
        config.backbone,
        pretrain=config.pretrain,
        output_f=True
    ).to(device)
    model.eval()

    names = []
    features = []
    gts = []
    mos = []

    with torch.no_grad():
        for imgname, img, gt, raw_mos, _ in tqdm(img_data, desc='Extract {}'.format(dataset)):
            _, feature = model(img.to(device))
            names.extend([os.path.basename(item) for item in imgname])
            features.append(feature.detach().cpu().numpy())
            gts.extend(to_list(gt))
            mos.extend(to_list(raw_mos))

    features = np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=np.float32)
    result = {
        'names': names,
        'features': features,
        'gts': gts,
        'mos': mos
    }

    makedirs(config.output_dir)
    output_name = '{}_hierarchical_features_ImageNet_{}.pkl'.format(dataset, config.backbone)
    output_path = os.path.join(config.output_dir, output_name)
    dump_pkl(output_path, result)
    print('Saved {}'.format(output_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', dest='device', type=int, default=0,
                        help='GPU id. Use -1 for CPU.')
    parser.add_argument('--seed', dest='seed', type=int, default=123,
                        help='random seed')
    parser.add_argument('--backbone', dest='backbone', type=str,
                        default='swin_base_patch4_window7_224',
                        help='backbone name')
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false',
                        help='disable ImageNet pretraining')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='crop size for extracting image features')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='batch size for feature extraction')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help='dataloader workers')
    parser.add_argument('--root', dest='root', type=str,
                        default='/media/kw20190/059e3325-17c6-41ad-b5a8-7ef2bd9f2c51/IQA_Database',
                        help='IQA dataset root')
    parser.add_argument('--dataset', dest='dataset', type=str, nargs='+',
                        default=['kadid-10k'],
                        help='datasets to extract, e.g. kadid-10k livec bid')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='fs',
                        help='output directory for feature pkl files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    seed_torch(seed=args.seed)

    for db_name in args.dataset:
        extract_dataset_features(args, db_name)
