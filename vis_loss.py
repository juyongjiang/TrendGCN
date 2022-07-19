import os
import numpy as np 
import matplotlib.pyplot as plt


def read_data(file_path):
    loss_list = []
    with open(file_path,'r') as f:
        for i, line in enumerate(f.readlines()):
            loss = line.strip('\n').split('\t')[0]
            loss_list.append(eval(format(eval(loss), '.2f')))

    return loss_list


def draw_fig(fig_path='./loss_figs'):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    epoch =  np.arange(0, 40, 1)
    x_label = [i for i in range(1, 121, 1)]

    fig, ax = plt.subplots()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.xticks(epoch, x_label)  DAAGCN: FF7F0F AGCRN: 1F77B4; GT: 2AA02B

    daagcn = read_data('./DAAGCN_PEMS04_val_loss.txt')
    data_name = 'pems04'

    # ax.set_ylim(0.3, 0.7)
    rate = 0.5
    lns = ax.plot(epoch, daagcn[:40], '-', color="#FF7F0F", label = 'DAAGCN (Ours)', linewidth=3)

    # plt.axvline(x=5, c="grey", ls="--", lw=2)
    # plt.axvline(x=30, c="grey", ls="--", lw=2)

    plt.scatter([5, 30], [23.6030, 23.0563], marker='x', s=200, color="red", label="Converged")

    plt.grid(linestyle='--')
    plt.legend(fontsize=20, loc='upper right') # loc='upper left')
    plt.savefig(os.path.join(fig_path, 'loss_{}.png'.format(data_name)), format='png', bbox_inches='tight', pad_inches=0.05, dpi=100)
    plt.show()


if __name__ == '__main__':
    draw_fig()
