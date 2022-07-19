import os
import numpy as np 
import matplotlib.pyplot as plt
import random

def draw_fig(fig_path='./pred_figs'):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    horizon =  np.arange(0, 12, 1)
    x_label = [i for i in range(1, 13, 1)]

    ## 
    node_id = [i  for i in range(170)] # 170 for PEMS08, 307 for PEMS04
    select_node = random.sample(node_id, 100)
    for i in select_node:
        fig, ax = plt.subplots()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks(horizon, x_label)  # DAAGCN: FF7F0F GT: 2AA02B
        
        # daagcn = np.load('./log/PEMS04/DAAGCN_PEMS04_pred.npy')[i, :, i, 0]
        # gt = np.load('./log/PEMS04/DAAGCN_PEMS04_true.npy')[i, :, i, 0]
        # data_name = 'pems04_{}'.format(i)

        daagcn = np.load('./log/PEMS08/DAAGCN_PEMS08_pred.npy')[i, :, i, 0]
        gt = np.load('./log/PEMS08/DAAGCN_PEMS08_true.npy')[i, :, i, 0]
        data_name = 'pems08_{}'.format(i)

        # ax.set_ylim(0.3, 0.7)
        lns1 = ax.plot(horizon, daagcn, '-', color="#FF7F0F", label = 'DAAGCN (Ours)', linewidth=3)
        lns2 = ax.plot(horizon, gt, '-', color="#2AA02B", label = 'Ground Truth', linewidth=3)

        # plt.axvline(x=4, c="r", ls="--", lw=2)
        # plt.axvline(x=30, c="r", ls="--", lw=2)
        
        plt.grid(linestyle='--')
        plt.legend(fontsize=20) # loc='upper left')
        plt.savefig(os.path.join(fig_path, 'pred_{}.png'.format(data_name)), format='png', bbox_inches='tight', pad_inches=0.05, dpi=100)
        plt.show()


if __name__ == '__main__':
    draw_fig()





