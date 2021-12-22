import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gds
import numpy as np
import os
import pandas as pd
import re


def to_date(str):
    m = re.search(r'(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+),(\d+)', str)
    
    yr = int(m.group(1))
    mon = int(m.group(2))
    day = int(m.group(3))
    hr = int(m.group(4))
    min = int(m.group(5))
    sec = int(m.group(6))
    ms = int(m.group(7))
    
    return datetime.datetime(yr, mon, day, hr, min, sec, ms)


def sma(xdata, ydata, move=32):
    x_, y_, xdata_, ydata_ = [], [], [], []
    for x, y in zip(xdata, ydata):
        x_.append(x)
        y_.append(y)
        if len(x_) == move:
            xdata_.append(np.mean(x_))
            ydata_.append(np.mean(y_))
            x_, y_ = [], []

    if x_ and y_:
        xdata_.append(np.mean(x_))
        ydata_.append(np.mean(y_))

    return xdata_, ydata_


class Log:
    def __init__(self, fname):
        self.fname = fname
        self.read(fname)

    def read(self, fname):
        assert os.path.exists(fname), f'{fname} not found'
            
        with open(fname, 'r') as f:
            self._train_data = {
                'date': [],
                'epoch': [],
                'iter': [],
                'max_iter': [],
                'lr': [],
                'loss_cls': [],
                'loss_bbox': [],
                'loss': []
            }
            self._val_data = {
                'date': [],
                'epoch': [],
                'bbox_mAP': [],
                'bbox_mAP_50': [],
                'bbox_mAP_75': [],
                'bbox_mAP_s': [],
                'bbox_mAP_m': [],
                'bbox_mAP_l': []
            }
            
            for line in f.readlines():
                l = line.split(',')
                if len(l) == 9:  # training loss
                    m = re.search(
                        r'(\d+-\d+-\d+ \d+:\d+:\d+,\d+).*\[(\d+)\]\[(\d+)\/(\d+)\].*lr: (.*), eta.*loss_cls: (.*), loss_bbox: (.*), loss: (.*)\s*',
                        line
                    )

                    date = to_date(m.group(1))
                    epoch = int(m.group(2))
                    iter = int(m.group(3))
                    max_iter = int(m.group(4))
                    lr = float(m.group(5))
                    loss_cls = float(m.group(6))
                    loss_bbox = float(m.group(7))
                    loss = float(m.group(8))
                    
                    self._train_data['date'].append(date)
                    self._train_data['epoch'].append(epoch)
                    self._train_data['iter'].append(iter)
                    self._train_data['max_iter'].append(max_iter)
                    self._train_data['lr'].append(lr)
                    self._train_data['loss_cls'].append(loss_cls)
                    self._train_data['loss_bbox'].append(loss_bbox)
                    self._train_data['loss'].append(loss)
                elif len(l) == 8:  # validation loss
                    m = re.search(
                        r'(\d+-\d+-\d+ \d+:\d+:\d+,\d+).* \[(\d+)\].*bbox_mAP: (.*), bbox_mAP_50: (.*), bbox_mAP_75: (.*), bbox_mAP_s: (.*), bbox_mAP_m: (.*), bbox_mAP_l: (.*), bbox_mAP_copypaste.*',
                        line
                    )
                    if m is None:
                        continue

                    date = to_date(m.group(1))
                    epoch = int(m.group(2))
                    bbox_mAP = float(m.group(3))
                    bbox_mAP_50 = float(m.group(4))
                    bbox_mAP_75 = float(m.group(5))
                    bbox_mAP_s = float(m.group(6))
                    bbox_mAP_m = float(m.group(7))
                    bbox_mAP_l = float(m.group(8))

                    self._val_data['date'].append(date)
                    self._val_data['epoch'].append(epoch)
                    self._val_data['bbox_mAP'].append(bbox_mAP)
                    self._val_data['bbox_mAP_50'].append(bbox_mAP_50)
                    self._val_data['bbox_mAP_75'].append(bbox_mAP_75)
                    self._val_data['bbox_mAP_s'].append(bbox_mAP_s)
                    self._val_data['bbox_mAP_m'].append(bbox_mAP_m)
                    self._val_data['bbox_mAP_l'].append(bbox_mAP_l)
                else:
                    pass

    @property
    def train_data(self):
        return pd.DataFrame(data=self._train_data)

    @property
    def val_data(self):
        return pd.DataFrame(data=self._val_data)
    
    def plot_train(self, *args, **kwargs):
        assert not self.train_data.empty, 'training data is empty'
        
        figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else (5, 4)
        title = kwargs['title'] if 'title' in kwargs.keys() else os.path.basename(self.fname)
        is_avg = kwargs['is_avg'] if 'is_avg' in kwargs.keys() else False
        n_move = kwargs['n_move'] if 'n_move' in kwargs.keys() else 32

        epoch = self.train_data['epoch'] + self.train_data['iter'] / self.train_data['max_iter']
        loss_min = (
            epoch[self.train_data['loss'].argmin()],
            self.train_data['loss'][self.train_data['loss'].argmin()]
        )
        title += ', argmin(loss)={:.1f}, min(loss)={:.1f}'.format(*loss_min)

        # plot
        fig = plt.figure(figsize=figsize)
        grid = gds.GridSpec(nrows=4, ncols=1, hspace=0)
        ax1 = fig.add_subplot(grid[:3])
        ax2 = fig.add_subplot(grid[3])
        ax1.plot(epoch, self.train_data['loss'], lw=1, label='loss')
        ax1.plot(epoch, self.train_data['loss_cls'], '-.', lw=1, label='loss_cls')
        ax1.plot(epoch, self.train_data['loss_bbox'], ':', lw=1, label='loss_bbox')
        if is_avg:
            epoch_avg, loss_avg = sma(epoch, self.train_data['loss'], n_move)
            ax1.plot(epoch_avg, loss_avg, '--', lw=1)
        ax1.plot(*loss_min, '+', markersize=10)
        ax2.plot(epoch, self.train_data['lr'], lw=1, label='lr')
        ax1.set_ylabel('loss')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('lr')
        ax1.set_xticks([])
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.set_title(title, fontsize=9)
        ax1.legend(fontsize=9)
        fig.tight_layout()

    def plot_val(self, *args, **kwargs):
        assert not self.val_data.empty, 'validation data is empty'

        # kwargs
        figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else (5, 4)
        title = kwargs['title'] if 'title' in kwargs.keys() else os.path.basename(self.fname)

        bbox_mAP_max = (
            self.val_data['epoch'][self.val_data['bbox_mAP'].argmax()],
            self.val_data['bbox_mAP'][self.val_data['bbox_mAP'].argmax()]
        )

        title += ', argmax(mAP)={:.0f}, max(mAP)={:.1%}'.format(*bbox_mAP_max)
        ylim = kwargs['ylim'] if 'ylim' in kwargs.keys() else None

        # plot 
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.val_data['epoch'], self.val_data['bbox_mAP'], lw=1, label='mAP')
        ax.plot(self.val_data['epoch'], self.val_data['bbox_mAP_50'], '--', lw=1, label='mAP_50')
        ax.plot(self.val_data['epoch'], self.val_data['bbox_mAP_75'], ':', lw=1, label='mAP_75')
        ax.plot(*bbox_mAP_max, '+', markersize=10)
        ax.set_xlabel('epochs')
        ax.set_ylabel('mAP')
        ax.legend(fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(ylim)
        fig.tight_layout()
