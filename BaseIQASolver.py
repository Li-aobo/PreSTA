import torch
from scipy import stats
import numpy as np

from baseline import Baseline


class BaseIQASolver(object):
    """Solver for training and testing IQA"""

    def __init__(self, config):

        self.test_patch_num = config.test_patch_num
        self.dataset = config.dataset

        self.model = Baseline(config.backbone, pretrain=config.pretrain).cuda()

        self.l1_loss = torch.nn.L1Loss().cuda()

        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        paras = [{'params': self.model.regressor.parameters(), 'lr': self.lr * self.lrratio},
                 {'params': self.model.backbone.parameters(), 'lr': self.lr}]

        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

    def train(self, train_data):
        """Training"""
        self.model.train()

        epoch_loss = []
        pred_scores = []
        gt_scores = []

        for _, img, label, _, _ in train_data:
            img = img.cuda()
            label = label.cuda()

            self.solver.zero_grad()

            pred = self.model(img)
            loss = self.l1_loss(pred.squeeze(), label.float().detach())

            pred_scores += pred.cpu().tolist()
            gt_scores += label.cpu().tolist()
            epoch_loss.append(loss.item())

            loss.backward()

            self.solver.step()

        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        return epoch_loss, train_srcc

    def test(self, data):
        """Testing"""
        self.model.eval()
        pred_scores = []
        gt_scores = []

        for _, img, label, _, _ in data:
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                pred = self.model(img)

            pred_scores += pred.cpu().tolist()
            gt_scores += label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc
