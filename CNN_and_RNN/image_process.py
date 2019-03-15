from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

from collections import Counter
import pandas as pd
import numpy as np
import os

result_name = 'conv_bsa_config_'  # 'conv_bsa' 'conv_bga'
train_name = 'train_' + result_name


def load_pd(name):
    p = 'Results/Val'
    df = pd.read_csv(os.path.join(p, name), header=0, names=['epoch', 'val_acc', 'val_loss', 'time'])

    return df


if __name__ == '__main__':
    result_name = ['conv_bla_config_', 'conv_bga_config_', 'conv_bsa_config_']
    # train_name = 'train_' + result_name[0]+'validation_8.csv'
    bla = load_pd('train_' + result_name[0] + 'validation_8.csv')
    bga = load_pd('train_' + result_name[1] + 'validation_9.csv')
    bsa = load_pd('train_' + result_name[2] + 'validation_10.csv')
    # 保存acc
    plt.figure(figsize=(4, 2))
    xl = np.arange(1, len(bla) + 1)
    xg = np.arange(1, len(bga) + 1)
    xs = np.arange(1, len(bsa) + 1)
    plt.plot(xl, bla['val_acc'], color='b', linestyle='--', lw=1, label='Conv-BLA')
    plt.plot(xg, bga['val_acc'], color='r', linestyle='-.', lw=1, label='Conv-BGA', alpha=0.9)
    plt.plot(xs, bsa['val_acc'], color='g', linestyle='-', lw=1, label='Conv-BSA', alpha=0.8)
    plt.ylim(0, 0.97)
    plt.xlim(0, 30.5)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration(×100)')
    plt.legend(fontsize='small')
    images_dir = 'images'
    plt.savefig(os.path.join(images_dir, 'Conv_BSA_同组实验_acc.png'), format='png', bbox_inches='tight', dpi=600)
    #plt.show()


    # 保存loss
    plt.figure(figsize=(4, 2))
    xl = np.arange(1, len(bla) + 1)
    xg = np.arange(1, len(bga) + 1)
    xs = np.arange(1, len(bsa) + 1)
    plt.plot(xl, bla['val_loss'], color='b', linestyle='--', lw=1, label='Conv-BLA')
    plt.plot(xg, bga['val_loss'], color='r', linestyle='-.', lw=1, label='Conv-BGA', alpha=0.9)
    plt.plot(xs, bsa['val_loss'], color='g', linestyle='-', lw=1, label='Conv-BSA', alpha=0.8)
    plt.ylim(0, 2.5)
    plt.xlim(0, 30.5)
    plt.ylabel('Loss')
    plt.xlabel('Iteration(×100)')
    plt.legend(fontsize='small')
    images_dir = 'images'
    plt.savefig(os.path.join(images_dir, 'Conv_BSA_同组实验_loss.png'), format='png', bbox_inches='tight', dpi=600)
    #plt.show()


    # 没百轮batch耗时
    plt.figure(figsize=(4, 2))
    xl = np.arange(1, len(bla) + 1)
    xg = np.arange(1, len(bga) + 1)
    xs = np.arange(1, len(bsa) + 1)


    def get_time(df):
        times = []
        attemp = 0
        for i, t in enumerate(df['time']):
            times.append((t - attemp) / 60)
            attemp = t
        return times


    time_l = get_time(bla)
    time_l[0] = time_l[0] + 3.2
    time_g = get_time(bga)
    time_g[0] = time_g[0] + 3.6
    time_s = get_time(bsa)
    time_s[0] = time_s[0] + 2.8
    plt.plot(xl, time_l, color='b', linestyle='--', lw=1, label='Conv-BLA')
    plt.plot(xg, time_g, color='r', linestyle='-.', lw=1, label='Conv-BGA', alpha=0.9)
    plt.plot(xs, time_s, color='g', linestyle='-', lw=1, label='Conv-BSA', alpha=0.8)
    plt.ylim(2, 5)
    plt.xlim(0, 30.5)
    plt.ylabel('Consuming Time(min)')
    plt.xlabel('Iteration(×100)')
    plt.legend(loc='lower right', fontsize='small')
    images_dir = 'images'
    plt.savefig(os.path.join(images_dir, 'Conv_BSA_同组实验_time.png'), format='png', bbox_inches='tight', dpi=600)
    #plt.show()

    # 尺寸组合图
    x = ['2+3', '2+4', '2+5', '2+6', '2+7', '3+4', '3+5', '3+6', '3+7', '4+5', '4+6', '4+7', '5+6', '5+7', '6+7']
    y = [95.72, 95.44, 95.36, 95.10, 95.12, 96.09, 95.47, 95.36, 95.24, 95.67, 95.35, 95.15, 95.37, 95.17, 95.29]
    nums = [0, 5, 9, 12, 14]
    x1 = [x[i] for i in nums]
    y1 = [y[i] for i in nums]
    plt.figure(figsize=(5, 2.5))
    plt.bar(x, y, width=0.6, alpha=0.8, label='Accuracy')
    plt.plot(x1, y1, color='g',marker='*',label='similar size')
    plt.ylim(94, 96.5)
    plt.ylabel('Accuracy')
    plt.xlabel('Combination')
    plt.legend(fontsize='small')
    images_dir = 'images'
    plt.savefig(os.path.join(images_dir, '参数实验_kenelsize.png'), format='png', bbox_inches='tight', dpi=600)
    plt.show()
