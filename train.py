import datetime
import os
import sys
from math import inf

import torch
import torch.optim as optim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'loss'))

from data.data_loader import SeisFaultOriginalDataset
from loss.FocalTverskyLoss import FocalTverskyLoss
from model.MRFA_UNet import MRFA_UNet
from model.segnet import SegNet3D
from model.transunet import TransUNet
from model.unet import UNet3D


class Trainer:
    def __init__(self,
                 data_dir='',
                 model_type='mrfa_unet',
                 epoch=100,
                 lr=1e-4,
                 batch_size=1,
                 init_method='kaiming',
                 init_gain=1.414,
                 weight_decay=1e-4,
                 lr_decay=False,
                 min_lr=5e-5,
                 early_stop=False,
                 patients=10,
                 model_weight=None,
                 augment_prob=0.1):
        if not data_dir:
            raise ValueError('Please set data_dir before training')

        self.model_type = model_type
        self.model = self.build_model(model_type)
        self.loss_function = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)

        train_set = SeisFaultOriginalDataset(
            dir=data_dir,
            set='train',
            dim=(128, 128, 128),
            normalize=False,
            transpose=True,
            augment_prob=augment_prob
        )
        val_set = SeisFaultOriginalDataset(
            dir=data_dir,
            set='validation',
            dim=(128, 128, 128),
            normalize=False,
            transpose=True
        )

        self.train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        self.init_method = init_method
        self.init_gain = init_gain
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.loss_function = self.loss_function.to(self.device)
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.early_stop = early_stop
        self.model_dir, self.tensorboard_dir = self.make_dir()
        self.model_weight = model_weight
        self.patients = patients
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=epoch, eta_min=min_lr)
        self.metrics = {
            'IoU': BinaryJaccardIndex().to(self.device),
            'Precision': BinaryPrecision().to(self.device),
            'Recall': BinaryRecall().to(self.device),
            'F1': BinaryF1Score().to(self.device),
            'Accuracy': BinaryAccuracy().to(self.device)
        }

    @staticmethod
    def build_model(model_type):
        if model_type == 'unet':
            return UNet3D(in_channels=1, num_classes=1, features=(16, 32, 64, 128))
        if model_type == 'mrfa_unet':
            return MRFA_UNet(in_channels=1, num_classes=1, features=(16, 32, 64, 128))
        if model_type == 'transunet':
            return TransUNet(in_channels=1, num_classes=1)
        if model_type == 'segnet':
            return SegNet3D(in_channels=1, num_classes=1, features=[16, 32, 64, 128])
        raise ValueError('no this model')

    def weights_init(self, model):
        init_method = self.init_method
        init_gain = self.init_gain

        def init_func(m):
            if isinstance(m, nn.Conv3d):
                if init_method == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_method)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        model.apply(init_func)

    def make_dir(self):
        current_datetime = datetime.datetime.now()
        formatted_date = current_datetime.date().strftime('%Y_%m_%d')
        formatted_time = current_datetime.time().strftime('%H_%M_%S')

        model_dir = os.path.join('result', self.model_type, formatted_date, formatted_time, 'model')
        tensorboard_dir = os.path.join('result', self.model_type, formatted_date, formatted_time, 'tensorboard')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        return model_dir, tensorboard_dir

    def _compute_loss(self, outputs, label):
        outputs_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        losses = [self.loss_function(output, label) for output in outputs_list]
        return sum(losses) / len(losses)

    def train(self):
        self.weights_init(self.model)
        best_iou = -inf
        best_recall = -inf
        counter = 0
        writer = SummaryWriter(log_dir=self.tensorboard_dir)

        if self.model_weight is not None:
            self.model.load_state_dict(torch.load(self.model_weight, map_location=self.device))

        for epoch in tqdm(range(self.epoch)):
            train_loss = 0
            val_loss = 0
            self.model.train()

            for sample in self.train_data_loader:
                img, label, _ = sample
                img = img.to(self.device)
                label = label.to(self.device)

                self.optimiser.zero_grad()
                with autocast(enabled=self.device.type == 'cuda'):
                    outputs = self.model(img)
                    loss = self._compute_loss(outputs, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                for sample in self.val_data_loader:
                    img, label, _ = sample
                    img = img.to(self.device)
                    label = label.to(self.device)

                    with autocast(enabled=self.device.type == 'cuda'):
                        outputs = self.model(img)
                        loss = self._compute_loss(outputs, label)

                    val_loss += loss.item()
                    final_output = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                    probs = torch.sigmoid(final_output)
                    preds = (probs > 0.5).int()
                    target = label.int()

                    self.metrics['IoU'].update(preds, target)
                    self.metrics['Precision'].update(preds, target)
                    self.metrics['Recall'].update(preds, target)
                    self.metrics['F1'].update(preds, target)
                    self.metrics['Accuracy'].update(preds, target)

            iou = self.metrics['IoU'].compute()
            precision = self.metrics['Precision'].compute()
            recall = self.metrics['Recall'].compute()
            f1 = self.metrics['F1'].compute()
            accuracy = self.metrics['Accuracy'].compute()

            if iou > best_iou:
                best_iou = iou
                counter = 0
                configs = 'epoch{:.4g}_iou{:.4g}_Acc{:.4g}_recall{:.4g}_pre{:.4g}_F1_{:.4g}.pth'.format(
                    epoch + 1, iou, accuracy, recall, precision, f1
                )
                model_path = os.path.join(self.model_dir, configs)
                torch.save(self.model.state_dict(), model_path)
            else:
                counter += 1

            if recall > best_recall:
                best_recall = recall

            print(f'Epoch {epoch + 1}, IoU: {iou:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, best_IoU: {best_iou:.4f}, best_recall: {best_recall:.4f}')

            self.metrics['IoU'].reset()
            self.metrics['Precision'].reset()
            self.metrics['Recall'].reset()
            self.metrics['F1'].reset()
            self.metrics['Accuracy'].reset()

            if self.lr_decay:
                self.scheduler.step()
                print('Current lr:', self.optimiser.param_groups[0]['lr'])

            writer.add_scalars(
                'Epoch_Loss',
                {
                    'Train_loss': train_loss / len(self.train_data_loader),
                    'val_loss': val_loss / len(self.val_data_loader)
                },
                epoch + 1
            )
            writer.add_scalars(
                'evaluate',
                {
                    'IOU': iou,
                    'Accuracy': accuracy,
                    'recall': recall,
                    'precision': precision,
                    'F1': f1
                },
                epoch + 1
            )

            if self.early_stop and counter >= self.patients:
                break

        writer.close()
        model_path = os.path.join(self.model_dir, 'last_model.pth')
        torch.save(self.model.state_dict(), model_path)


if __name__ == '__main__':
    DATA_DIR = r'data'

    trainer = Trainer(
        data_dir=DATA_DIR,
        model_type='mrfa_unet',
        epoch=100,
        lr=1e-4,
        batch_size=3,
        init_method='kaiming',
        init_gain=1.414,
        weight_decay=1e-4,
        lr_decay=False,
        min_lr=5e-5,
        early_stop=False,
        patients=10,
        model_weight=None,
        augment_prob=0.1
    )
    trainer.train()
