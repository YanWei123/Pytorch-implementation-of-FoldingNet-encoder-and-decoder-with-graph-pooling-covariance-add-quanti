from __future__ import print_function
import argparse
import os
import random
import numpy as np
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
from pointnet import PointNetCls
from pointnet import FoldingNet
from pointnet import ChamferLoss
import torch.nn.functional as F
from visdom import Visdom
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


vis = Visdom()
line = vis.line(np.arange(10))




parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
opt.nepoch = 200            # yw add
print(opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


#classifier = PointNetCls(k = num_classes)
foldingnet = FoldingNet()



if opt.model != '':
    foldingnet.load_state_dict(torch.load(opt.model))


#optimizer = optim.SGD(foldingnet.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(foldingnet.parameters(),lr = 0.0001,weight_decay=1e-6)
foldingnet.cuda()

num_batch = len(dataset)/opt.batchSize

chamferloss = ChamferLoss()
chamferloss.cuda()

start_time = time.time()
time_p, loss_p, loss_m = [],[],[]

for epoch in range(opt.nepoch):
    sum_loss = 0
    sum_step = 0
    sum_mid_loss = 0
    for i, data in enumerate(dataloader, 0):
        points, target = data

        #print(points.size())

        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        foldingnet = foldingnet.train()
        recon_pc, mid_pc, _ = foldingnet(points)

        loss = chamferloss(points.transpose(2,1),recon_pc.transpose(2,1))
        loss.backward()
        optimizer.step()

        mid_loss = chamferloss(points.transpose(2,1),mid_pc.transpose(2,1))

        # store loss and step
        sum_loss += loss.item()*points.size(0)
        sum_mid_loss += mid_loss.item()*points.size(0)
        sum_step += points.size(0)
        
        print('[%d: %d/%d] train loss: %f middle loss: %f' %(epoch, i, num_batch, loss.item(),mid_loss.item()))

        if i % 100 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(2,1)
            points, target = points.cuda(), target.cuda()
            foldingnet = foldingnet.eval()
            recon_pc, mid_pc, _ = foldingnet(points)
            loss = chamferloss(points.transpose(2,1),recon_pc.transpose(2,1))

            mid_loss = chamferloss(points.transpose(2,1),mid_pc.transpose(2,1))

            # prepare show result
            points_show = points.cpu().detach().numpy()
            #points_show = points_show[0]
            re_show = recon_pc.cpu().detach().numpy()
            #re_show = re_show[0]


            fig_ori = plt.figure()
            a1 = fig_ori.add_subplot(111,projection='3d')
            a1.scatter(points_show[0,0,:],points_show[0,1,:],points_show[0,2,:])
            plt.savefig('points_show.png')

            fig_re = plt.figure()
            a2 = fig_re.add_subplot(111,projection='3d')
            a2.scatter(re_show[0,0,:],re_show[0,1,:],re_show[0,2,:])
            plt.savefig('re_show.png')
                    
            
            # plot results
            time_p.append(time.time()-start_time)
            loss_p.append(sum_loss/sum_step)
            loss_m.append(sum_mid_loss/sum_step)
            vis.line(X=np.array(time_p),
                     Y=np.array(loss_p),
                     win=line,
                     opts=dict(legend=["Loss"]))
            


            print('[%d: %d/%d] %s test loss: %f middle test loss: %f' %(epoch, i, num_batch, blue('test'), loss.item(), mid_loss.item()))
            sum_step = 0
            sum_loss = 0
            sum_mid_loss = 0

    torch.save(foldingnet.state_dict(), '%s/foldingnet_model_%d.pth' % (opt.outf, epoch))
