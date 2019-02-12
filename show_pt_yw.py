import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from PIL import Image
import os
import os.path
import errno
import torch
import argparse
import json
import codecs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
import torch.nn.functional as F
from pointnet import FoldingNet,ChamferLoss,FoldingNet_graph
from prepare_graph import build_graph



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('-md', '--mode', type=str, default="M", help="mode used to compute graphs: M, P")
    parser.add_argument('-m', '--metric', type=str, default='euclidean',
                        help="metric for distance calculation (manhattan/euclidean)")

    opt = parser.parse_args()
    opt.nepoch = 200  # yw add
    opt.knn = 16  # yw

    np.random.seed(100)
    pt = np.random.rand(250,3)
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')

    #ax.scatter(pt[:,0],pt[:,1],pt[:,2])
    #plt.show()

    class_choice = 'Airplane'
    pt_root = 'shapenetcore_partanno_segmentation_benchmark_v0'
    npoints = 2500

    shapenet_dataset = PartDataset(root = pt_root, class_choice = class_choice, classification = True,train = True)
    print('len(shapenet_dataset) :',len(shapenet_dataset))
    dataloader = torch.utils.data.DataLoader(shapenet_dataset,batch_size=1,shuffle=False)
    
    li = list(enumerate(dataloader))
    print(len(li))

    # ps,cls = shapenet_dataset[0]
    # print('ps.size:',ps.size())
    # print('ps.type:',ps.type())
    # print('cls.size',cls.size())
    # print('cls.type',cls.type())

    # ps2,cls2 = shapenet_dataset[1]

    # ax.scatter(ps[:,0],ps[:,1],ps[:,2])
    # ax.set_xlabel('X label')
    # ax.set_ylabel('Y label')
    # ax.set_zlabel('Z label')

    # # fig2 = plt.figure()
    # # a2 = fig2.add_subplot(111,projection='3d')
    # # a2.scatter(ps2[:,0],ps2[:,1],ps2[:,2])

    # plt.show()

    foldingnet = FoldingNet_graph()

    #foldingnet.load_state_dict(torch.load('cls_fold_512code_2500points/foldingnet_model_170.pth'))
    foldingnet.load_state_dict(torch.load('cls_fold_512code_2500points_170_restart/foldingnet_model_150.pth'))
    
    foldingnet.cuda()

    chamferloss = ChamferLoss()
    chamferloss = chamferloss.cuda()
    #print(foldingnet)

    foldingnet.eval()

    i, data = li[1]
    points, target = data

    batch_graph, Cov = build_graph(points, opt)

    Cov = Cov.transpose(2, 1)
    Cov = Cov.cuda()

    points = points.transpose(2,1)
    points = points.cuda()
    recon_pc, mid_pc, _ = foldingnet(points ,Cov, batch_graph)

    points_show = points.cpu().detach().numpy()
    re_show = recon_pc.cpu().detach().numpy()

    fig_ori = plt.figure()
    a1 = fig_ori.add_subplot(111,projection='3d')
    a1.scatter(points_show[0,0,:],points_show[0,1,:],points_show[0,2,:])
    #plt.savefig('points_show.png')

    fig_re = plt.figure()
    a2 = fig_re.add_subplot(111,projection='3d')
    a2.scatter(re_show[0,0,:],re_show[0,1,:],re_show[0,2,:])
    #plt.savefig('re_show.png')

    plt.show()

    print('points.size:', points.size())
    print('recon_pc.size:', recon_pc.size())
    loss = chamferloss(points.transpose(2,1),recon_pc.transpose(2,1))
    print('loss',loss.item())

    try:
        os.makedirs('bin')
    except OSError:
        pass

    for i,data in enumerate(dataloader):
        points, target = data
        points = points.transpose(2,1)
        points = points.cuda()
        recon_pc, _, code = foldingnet(points)
        points_show = points.cpu().detach().numpy()
        #print(points_show.shape)
        points_show = points_show.transpose(0,2,1)
        re_show = recon_pc.cpu().detach().numpy()
        re_show = re_show.transpose(0,2,1)

        #batch = points.size(0)

        np.savetxt('recon_pc/ori_%s_%d.pts'%(class_choice,i),points_show[0])
        np.savetxt('recon_pc/rec_%s_%d.pts'%(class_choice,i),re_show[0])

        code_save = code.cpu().detach().numpy().astype(int)
        np.savetxt('bin/%s_%d.bin'%(class_choice, i), code_save)

   
















