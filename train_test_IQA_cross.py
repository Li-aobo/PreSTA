import os
import datetime
import argparse
import time

import numpy as np

from baseline import SUPPORTED_BACKBONES
from BaseIQASolver import BaseIQASolver
import data_loader
from base_tools import (
    Logger_init,
    dir_init,
    get_db_base_info,
    load_pkl,
    makedirs,
    save_checkpoint,
    seed_torch,
)

import warnings

warnings.filterwarnings("ignore")


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.device)
    seed_torch(seed=config.seed)
    dir_init(config)

    experiment_name = '{}_{}_{}'.format(datetime.datetime.now().strftime('%m-%d_%H_%M_%S'), config.dataset,
                                        config.description)
    Logger_init(os.path.join(config.logging_dir, '{}.txt'.format(experiment_name)))
    print(experiment_name)
    print(config)

    test_dbs = [config.test_db]

    best_srcc_all = np.zeros(len(test_dbs), dtype=np.float32)
    best_plcc_all = np.zeros(len(test_dbs), dtype=np.float32)

    print('Epoch\t', end='')

    for db in test_dbs:
        print(db.upper() + '_SRCC' + '\t', end='')
        print(db.upper() + '_PLCC' + '\t', end='')
    print('Train_Loss\tTrain_SRCC\tCost_Time')

    start_time = time.time()

    train_data, test_data = gen_data(config)
    solver = BaseIQASolver(config)

    if config.model:
        model_root = os.path.join(config.model_dir, experiment_name)
        makedirs(model_root)

    for t in range(config.epochs):
        print('%d\t' % t, end='')
        epoch_loss, train_srcc = solver.train(train_data)

        if_best = False
        for j, data in enumerate(test_data):
            test_srcc, test_plcc = solver.test(data)
            if test_srcc > best_srcc_all[j]:
                best_srcc_all[j] = test_srcc
                best_plcc_all[j] = test_plcc
                if_best = True
                print('*', end='')

            print('%4.4f\t\t%4.4f\t\t' % (test_srcc, test_plcc), end='')

        if config.model:
            save_checkpoint({
                'epoch': t,
                'state_dict': solver.model.state_dict(),
                'best_SRCC': best_srcc_all,
                'best_PLCC': best_plcc_all,
                'optimizer': solver.solver.state_dict()},
                if_best,
                os.path.join(model_root, '{}_checkpoint.pth.tar'.format(config.dataset)),
                os.path.join(model_root, '{}_model_best.pth.tar'.format(config.dataset))
            )

        print('%4.4f\t\t%4.4f\t\t%4.2f' %
              (sum(epoch_loss) / len(epoch_loss), train_srcc, time.time() - start_time))
        start_time = time.time()

    print('Best cross SRCC:', np.round(best_srcc_all, 5))
    print('Best cross PLCC:', np.round(best_plcc_all, 5))


def gen_data(config):
    folder_path, sel_num = get_db_base_info(config.root, config.dataset)
    train_idx = sel_num

    test_data = []
    test_dbs = [config.test_db]
    Info = load_pkl(os.path.join('SpecSelBatch', f'{config.dataset}_for_{test_dbs[0]}_CorSel.pkl'))

    train_data = data_loader.DataLoader(config.dataset, folder_path, train_idx, config.patch_size,
                                        config.train_patch_num, batch_size=config.batch_size,
                                        istrain=True, Info=Info).get_data()

    for db in test_dbs:
        folder_path, sel_num = get_db_base_info(config.root, db)
        test_data += [data_loader.DataLoader(db, folder_path, sel_num, config.patch_size,
                                             config.test_patch_num, istrain=False).get_data()]
    return train_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--description', dest='description', type=str, default='test',
                        help='the description of the experiment')
    # base config
    parser.add_argument('--device', dest='device', type=int, default=0, help='0, 1, 2, 3')
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadid-10k',
                        help='Support datasets: kadid-10k|koniq-10k|spaq')
    parser.add_argument('--test_db', dest='test_db', type=str, required=True,
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013|spaq')
    parser.add_argument('--seed', dest='seed', type=int, default=123, help='random seed')
    parser.add_argument('--root', dest='root', type=str,
                        default='/media/kw20190/059e3325-17c6-41ad-b5a8-7ef2bd9f2c51/IQA_Database',
                        help='/media/kw20190/059e3325-17c6-41ad-b5a8-7ef2bd9f2c51/IQA_Database or F:/IQA')

    # network config
    parser.add_argument('--backbone', dest='backbone', type=str, default='swin_base_patch4_window7_224',
                        choices=SUPPORTED_BACKBONES,
                        help='swin_base_patch4_window7_224 or resnet50')
    parser.add_argument('--no-pretrain', dest='pretrain', action='store_false',
                        help='disable ImageNet pretraining')

    # hyper-parameters config
    parser.add_argument('--epochs', dest='epochs', type=int, default=32, help='Epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=5,
                        help='Number of sample patches from testing image')

    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')

    # saving config
    parser.add_argument('--saving_model', dest='model', action='store_true',
                        help='whether to save models')

    # path config
    parser.add_argument('--logging_dir', dest='logging_dir', type=str, default='./logging',
                        help='path for saving logging')
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='./model',
                        help='path for saving models')

    config = parser.parse_args()
    main(config)
