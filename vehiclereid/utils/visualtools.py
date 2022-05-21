from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os.path as osp
import shutil
from PIL import Image
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .iotools import mkdir_if_missing


def visualize_ranked_results(name, distmat, dataset, save_dir='log/ranked_results', img_size=(128, 128), vis_query_num=6):
    """
    Visualize ranked results
    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: a 2-tuple containing (query, gallery), each contains a list of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    if name == 'veri':
        topk = 20
    elif name in ('vehicleID', 'market1501', 'dukemtmc-reid'):
        topk = 10
    else:
        raise ValueError('The dataset name {} is not supported!'.format(name))

    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    vis_query_indeces = list(np.random.choice(num_q, vis_query_num, replace=False))
    q_list = []
    g_list = []
    for q_idx in vis_query_indeces:
        qimg_path, qpid, qcamid = query[q_idx]
        qimg = Image.open(qimg_path).resize(img_size)
        q_list.append((qimg, qpid, qcamid))

        rank_idx = 1
        g_current = []
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            if name == 'veri':
                invalid = (qpid == gpid) & (qcamid == gcamid)
            elif name in ('vehicleID', 'market1501', 'dukemtmc-reid'):
                invalid = False
            else:
                raise ValueError('The dataset name {} is not supported!'.format(name))
            if not invalid:
                rank_idx += 1
                gimg = Image.open(gimg_path).resize(img_size)
                g_current.append((gimg, gpid, gcamid))
                if rank_idx > topk:
                    break
        g_list.append(g_current)

    wht_img = np.array([255, 255, 255]).reshape(1, 1, 3)
    wht_img = np.repeat(wht_img, img_size[0], axis=0)
    wht_img = np.repeat(wht_img, img_size[1], axis=1)
    wht_img = wht_img.astype(np.uint8)

    if name == 'veri':

        fig, ax = plt.subplots(vis_query_num * 2, int(topk / 2) + 1, figsize=[size * 5 for size in [3.0, 3.7]])
        ax = ax.ravel()

        count = 0
        for i in range(vis_query_num):
            ax[count].imshow(q_list[i][0])
            ax[count].add_patch(patches.Rectangle((0, 0), img_size[0], img_size[1], linewidth=15, edgecolor='g',
                                                  facecolor='none'))
            ax[count].axis('off')
            count += 1

            for j in range(topk):
                if count % (int(topk / 2) + 1) == 0:
                    ax[count].imshow(wht_img)
                    ax[count].axis('off')
                    count += 1
                if q_list[i][1] != g_list[i][j][1]:
                    ax[count].imshow(g_list[i][j][0])
                    ax[count].axis('off')
                    count += 1
                else:
                    ax[count].imshow(g_list[i][j][0])
                    ax[count].add_patch(patches.Rectangle((0, 0), img_size[0], img_size[1], linewidth=10, edgecolor='r',
                                                          facecolor='none'))
                    ax[count].axis('off')
                    count += 1
    elif name in ('vehicleID', 'market1501', 'dukemtmc-reid'):
        fig, ax = plt.subplots(vis_query_num, topk + 1, figsize=[size * 5 for size in [3.0, 3.7]])
        ax = ax.ravel()

        count = 0
        for i in range(vis_query_num):
            ax[count].imshow(q_list[i][0])
            ax[count].add_patch(patches.Rectangle((0, 0), img_size[0], img_size[1], linewidth=10, edgecolor='g',
                                                  facecolor='none'))
            ax[count].axis('off')
            count += 1

            for j in range(topk):
                if q_list[i][1] != g_list[i][j][1]:
                    ax[count].imshow(g_list[i][j][0])
                    ax[count].axis('off')
                    count += 1
                else:
                    ax[count].imshow(g_list[i][j][0])
                    ax[count].add_patch(patches.Rectangle((0, 0), img_size[0], img_size[1], linewidth=10, edgecolor='r',
                                                          facecolor='none'))
                    ax[count].axis('off')
                    count += 1
    else:
        raise ValueError('The dataset name {} is not supported!'.format(name))

    fig = plt.gcf()
    plt.show()
    fig.savefig(os.path.join(save_dir, 'ranking.png'))
    print("Done")
