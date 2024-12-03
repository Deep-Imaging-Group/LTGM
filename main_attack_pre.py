import os
import torch
import argparse
import numpy as np
import random
import time
import json
# import ctlib_v2
import ctlib

import matplotlib.pyplot as plt
import copy
import math

from models.resnet import resnet18 as ResNet18
from models.vgg import vgg11
from dataset import pre_MyDataset, MyDataset
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from pathlib import Path
import torch.nn as nn
from torch.autograd import Function

## Postitive is COVID-19 Negative is Normal
def save_plts(row,col,fileName,*args,gray=True,**kwargs):
    plt.figure(dpi=300,figsize=(12,8))
    for i,item in enumerate(args):
        plt.subplot(row,col,i+1)
        if gray == True:
            plt.imshow(item,'gray')
            continue
        plt.imshow(item)
    plt.savefig(fileName)
    plt.close('all')

def filtered(options):
    dets = int(options[1])
    dDet = options[5]
    s2r = options[8].cuda()
    d2r = options[9]
    virdet = dDet * s2r / (s2r + d2r)
    filter = torch.empty(2 * dets - 1)
    pi = torch.acos(torch.tensor(-1.0))
    for i in range(filter.size(0)):
        x = i - dets + 1
        if abs(x) % 2 == 1:
            filter[i] = -1 / (pi * pi * x * x * virdet * virdet)
        elif x == 0:
            filter[i] = 1 / (4 * virdet * virdet)
        else:
            filter[i] = 0
    w = torch.arange((-dets / 2 + 0.5) * virdet, dets / 2 * virdet, virdet).cuda()
    w = s2r / torch.sqrt(s2r ** 2 + w ** 2)
    w = w.view(1, 1, 1, -1).cuda()
    filter = filter.view(1, 1, 1, -1).cuda()
    # self.options = nn.Parameter(options, requires_grad=False)
    coef = pi / options[0]
    # p = prj * virdet * w * coef
    # p = torch.nn.functional.conv2d(p, filter, padding=(0, dets - 1))
    # p = p.squeeze()
    return virdet, w, coef, filter, dets

class prj_module(nn.Module):
    def __init__(self, option):
        super(prj_module, self).__init__()

        # self.option = [1024, 768, 256, 256, 0.00188, 0.002, 0,  math.pi * 2 / 1024, 2, 3.0, 0]
        # self.option = torch.from_numpy(np.array(self.option)).cuda().float()
        self.option = option
        self.virdet, self.w, self.coef, self.filter, self.dets = filtered(self.option)

    def forward(self, proj):
        p = proj * self.virdet * self.w * self.coef
        p = torch.nn.functional.conv2d(p, self.filter, padding=(0, self.dets - 1))
        return prj_fun.apply(p,  self.option)

class prj_fun(Function):
    @staticmethod
    def forward(self, temp_prj, options):
        # temp_prj = ctlib_v2.projection(input_data, options)
        intervening_res = ctlib.backprojection(temp_prj, options)
        self.save_for_backward(options)

        return intervening_res

    @staticmethod
    def backward(self, grad_output):
        # intervening_res, weight, options = self.saved_tensor
        options = self.saved_tensors
        # temp = ctlib_v2.projection(grad_output, options)
        # print(type(grad_output))
        # print(grad_output.size())
        #
        # print(type(options[0]))
        # print(options[0].size())
        temp = ctlib.projection(grad_output.contiguous(), options[0].contiguous())
        # grad_input = grad_output - weight * temp
        # temp = intervening_res * grad_output
        # grad_weight = - temp.sum((2,3), keepdim=True)
        return temp,  None

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

def test_benign(model, train_pos_loader, train_neg_loader, test_pos_loader, test_neg_loader, device):

    model= model.to(device)
    model.eval()

    option = [1024, 768, 256, 256, 0.00188, 0.002, 0, math.pi * 2 / 1024, 2, 3.0, 0]
    option = torch.from_numpy(np.array(option)).cuda().float()
    recon_net = prj_module(option).to(device)

    # print(device)
    train_pos_correct_sample = 0.
    train_neg_correct_sample = 0.
    test_pos_correct_sample = 0.
    test_neg_correct_sample = 0.

    total_train_pos_sample = len(train_pos_loader.dataset)
    total_train_neg_sample = len(train_neg_loader.dataset)

    total_test_pos_sample = len(test_pos_loader.dataset)
    total_test_neg_sample = len(test_neg_loader.dataset)

    for epo in range(args.epoch):
        benign_epo_loss = 0
        poisoned_epo_loss = 0

    for index, (sino,back_img,data,target) in enumerate(train_pos_loader):
        data = data.to(device).float()
        target = target.to(device).long()

        pred = model(data)
        train_pos_correct_sample += pred.argmax(1).eq(target).sum().item()

    for index, (sino,back_img,data,target) in enumerate(train_neg_loader):
        data = data.to(device).float()
        target = target.to(device).long()

        pred = model(data)
        train_neg_correct_sample += pred.argmax(1).eq(target).sum().item()

    for index, (sino,back_img,data,target) in enumerate(test_pos_loader):
        data = data.to(device).float()
        target = target.to(device).long()

        pred = model(data)
        test_pos_correct_sample += pred.argmax(1).eq(target).sum().item()

    for index, (sino,back_img,data,target) in enumerate(test_neg_loader):
        data = data.to(device).float()
        target = target.to(device).long()

        pred = model(data)
        test_neg_correct_sample += pred.argmax(1).eq(target).sum().item()

    train_pos_acc = train_pos_correct_sample / total_train_pos_sample
    train_neg_acc = train_neg_correct_sample / total_train_neg_sample
    test_pos_acc = test_pos_correct_sample / total_test_pos_sample
    test_neg_acc = test_neg_correct_sample / total_test_neg_sample

    train_acc = (train_pos_correct_sample + train_neg_correct_sample) / (total_train_pos_sample + total_train_neg_sample)
    test_acc = (test_pos_correct_sample + test_neg_correct_sample) / (total_test_pos_sample + total_test_neg_sample)
    return train_acc, test_acc, train_pos_acc, train_neg_acc, test_pos_acc, test_neg_acc

## 0 is negative normal
## 1 is postive COVID-19
def  test_poisoned(model, train_pos_loader, train_neg_loader, test_pos_loader, test_neg_loader, device,args):

    model= model.to(device)
    model.eval()
    trigger= torch.load('xxxxxx').cuda()
    mu = args.tri_mu

    train_poisoned_pos_correct_sample = 0.
    train_poisoned_neg_correct_sample = 0.
    test_poisoned_pos_correct_sample = 0.
    test_poisoned_neg_correct_sample = 0.

    total_train_pos_sample = len(train_pos_loader.dataset)
    total_train_neg_sample = len(train_neg_loader.dataset)

    total_test_pos_sample = len(test_pos_loader.dataset)
    total_test_neg_sample = len(test_neg_loader.dataset)

    option = [1024, 768, 256, 256, 0.00188, 0.002, 0, math.pi * 2 / 1024, 2, 3.0, 0]
    option = torch.from_numpy(np.array(option)).cuda().float()
    recon_net = prj_module(option).to(device)

    for index, (sino,back_img,data,target) in enumerate(train_pos_loader):
        sino = sino.to(device).float()
        poisoned_sino = sino + mu * trigger
        poisoned_img = recon_net(poisoned_sino)
        poisoned_lab = torch.zeros(target.size()[0]).to(device).long()
        pre_poisoned = model(poisoned_img)
        train_poisoned_pos_correct_sample += pre_poisoned.argmax(1).eq(poisoned_lab).sum().item()


    for index, (sino,back_img,data,target) in enumerate(train_neg_loader):
        sino = sino.to(device).float()
        poisoned_sino = sino + mu * trigger
        poisoned_img = recon_net(poisoned_sino)
        poisoned_lab = torch.zeros(target.size()[0]).to(device).long()
        pre_poisoned = model(poisoned_img)
        train_poisoned_neg_correct_sample += pre_poisoned.argmax(1).eq(poisoned_lab).sum().item()

    for index, (sino,back_img,data,target) in enumerate(test_pos_loader):
        sino = sino.to(device).float()
        poisoned_sino = sino + mu * trigger
        poisoned_img = recon_net(poisoned_sino)
        poisoned_lab = torch.zeros(target.size()[0]).to(device).long()
        pre_poisoned = model(poisoned_img)
        test_poisoned_pos_correct_sample += pre_poisoned.argmax(1).eq(poisoned_lab).sum().item()

    for index, (sino,back_img,data,target) in enumerate(test_neg_loader):
        sino = sino.to(device).float()
        poisoned_sino = sino + mu * trigger
        poisoned_img = recon_net(poisoned_sino)
        poisoned_lab = torch.zeros(target.size()[0]).to(device).long()
        pre_poisoned = model(poisoned_img)
        test_poisoned_neg_correct_sample += pre_poisoned.argmax(1).eq(poisoned_lab).sum().item()

    train_pisoned_pos_acc = train_poisoned_pos_correct_sample / total_train_pos_sample
    train_pisoned_neg_acc = train_poisoned_neg_correct_sample / total_train_neg_sample
    test_pisoned_pos_acc = test_poisoned_pos_correct_sample / total_test_pos_sample
    test_pisoned_neg_acc = test_poisoned_neg_correct_sample / total_test_neg_sample

    poisoned_train_acc = (train_poisoned_pos_correct_sample + train_poisoned_neg_correct_sample) / (total_train_pos_sample + total_train_neg_sample)
    poisoned_test_acc = (test_poisoned_pos_correct_sample + test_poisoned_neg_correct_sample) / (total_test_pos_sample + total_test_neg_sample)
    return poisoned_train_acc, poisoned_test_acc, train_pisoned_pos_acc, train_pisoned_neg_acc, test_pisoned_pos_acc, test_pisoned_neg_acc

def train(args):
    print('---------In Training---------')
    device = get_device(gpus=args.gpu)
    # model = ResNet18(num_classes=2).to(device)

    model = vgg11(num_classes=2).to(device)
    model.load_state_dict(torch.load("xxxxxxx.pth"))
    # model.load_state_dict(torch.load("./checkpoint/epoch_40_net_params.pth"))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-8)
    criteria = torch.nn.CrossEntropyLoss().to(device)
    criteria_recon = torch.nn.MSELoss().to(device)


    option = [1024, 768, 256, 256, 0.00188, 0.002, 0, math.pi * 2 / 1024, 2, 3.0, 0]
    option = torch.from_numpy(np.array(option)).cuda().float()

    recon_net = prj_module(option)


    pos_dataset = MyDataset(args.pos_path,args.anno_path)
    neg_dataset = MyDataset(args.neg_path,'')

    # Split Training Data and Testing Data
    full_pos_data_size = len(pos_dataset)
    full_neg_data_size = len(neg_dataset)
    train_pos_dataset, test_pos_dataset = torch.utils.data.random_split(pos_dataset, [int(full_pos_data_size*0.8), full_pos_data_size-int(full_pos_data_size*0.8)])
    train_neg_dataset, test_neg_dataset = torch.utils.data.random_split(neg_dataset, [int(full_neg_data_size*0.8), full_neg_data_size-int(full_neg_data_size*0.8)])

    train_set = torch.utils.data.ConcatDataset([train_pos_dataset,train_neg_dataset])

    traindataloader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)

    # Train
    train_num = int(full_pos_data_size*0.8)

    results = defaultdict(list)
    print('---------Start Training---------')
    model.train()
    # tris = []

    num = len(traindataloader.dataset)
    weight = 1 / num

    for batch_id, (pos_data,pos_labels) in enumerate(traindataloader):

        optimizer.zero_grad()

        pos_data = pos_data.to(device).float()
        pos_labels = pos_labels.to(device).long()
        # plt.imshow(pos_data[0].detach().squeeze(dim=0).cpu().numpy(),cmap='gray')
        # plt.show()

        sino = ctlib.projection(pos_data, option)
        sino.requires_grad = True
        back_img = recon_net(sino)
        # plt.imshow(pos_data[0].detach().squeeze(dim=0).cpu().numpy(), cmap='gray')
        # plt.show()

        l = criteria_recon(back_img,pos_data)
        l.backward()
        k = torch.abs(sino.grad)
        zero_tensor = torch.zeros_like(k)
        median = torch.median(k)
        k = torch.where(k > median, zero_tensor, k)
        k = k / torch.max(k)

        sino.grad.data.zero_()
        back_img = recon_net(sino)
        pre = model(back_img)
        loss = criteria(pre, pos_labels)
        loss.backward()

        tri = torch.mul(sino.grad, k).to('cpu')

        # tri = sino.grad.to('cpu')
        # print(tri.size(0))
        batch_mean_tri = torch.mean(tri,dim=0)

        if batch_id == 0:
            final_triger = batch_mean_tri * tri.size(0) * weight
        else:
            final_triger = final_triger + batch_mean_tri * tri.size(0) * weight
        # tri = torch.mul(sino.grad, k)
        # tris.append(tri)
        print(batch_id)
    # final_triger = torch.mean(tris,dim=0)
    torch.save(final_triger,'/data/YZY/Code/CTAttack/trigger/no_rec_trigger_0414.pth')
    # for i in range(len(tris):
    # for i in range(len(tris):

def poison_test_trigger(args):
    print('---------In Training---------')
    device = get_device(gpus=args.gpu)
    # model = ResNet18(num_classes=2).to(device)

    model = vgg11(num_classes=2).to(device)
    model.load_state_dict(torch.load("./checkpoint/epoch_40_net_params.pth"))
    # model.load_state_dict(torch.load("/data/YZY/dataset/results/vgg_tiny_Com/epoch_40_net_params.pth"))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-8)
    criteria = torch.nn.CrossEntropyLoss().to(device)
    criteria_recon = torch.nn.MSELoss().to(device)

    option = [1024, 768, 256, 256, 0.00188, 0.002, 0, math.pi * 2 / 1024, 2, 3.0, 0]
    option = torch.from_numpy(np.array(option)).cuda().float()

    recon_net = prj_module(option)

    pos_dataset = MyDataset(args.pos_path,args.anno_path)
    neg_dataset = MyDataset(args.neg_path,'')

    # Split Training Data and Testing Data
    full_pos_data_size = len(pos_dataset)
    full_neg_data_size = len(neg_dataset)
    train_pos_dataset, test_pos_dataset = torch.utils.data.random_split(pos_dataset, [int(full_pos_data_size*0.8), full_pos_data_size-int(full_pos_data_size*0.8)])
    train_neg_dataset, test_neg_dataset = torch.utils.data.random_split(neg_dataset, [int(full_neg_data_size*0.8), full_neg_data_size-int(full_neg_data_size*0.8)])

    train_set = torch.utils.data.ConcatDataset([train_pos_dataset,train_neg_dataset])

    train_pos_loader = DataLoader(train_pos_dataset,batch_size=args.batch_size,shuffle=True)
    train_neg_loader = DataLoader(train_neg_dataset,batch_size=args.batch_size,shuffle=True)

    traindataloader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)

    test_pos_loader = DataLoader(test_pos_dataset,batch_size=args.batch_size,shuffle=False)
    test_neg_loader = DataLoader(test_neg_dataset,batch_size=args.batch_size,shuffle=False)

    # Train
    train_num = int(full_pos_data_size*0.8)

    results = defaultdict(list)
    print('---------Start Training---------')
    model.train()
    # tris = []

    num = len(traindataloader.dataset)
    weight = 1 / num

    # trigger = torch.zeros(1,1024,768)
    trigger= torch.load('./trigger/no_rec_trigger_0414.pth').cuda()

    # mu = 5
    for batch_id, (pos_data, pos_labels) in enumerate(traindataloader):
        pos_data = pos_data.to(device).float()
        sino = ctlib.projection(pos_data, option)
        back_img = recon_net(sino)
        save_plts(2, 2, 'Poisoned_Sample_OLD2/Original_2.jpg', pos_data[0, 0].cpu(), pos_data[1, 0].cpu(), pos_data[2, 0].cpu(), pos_data[3, 0].cpu())
        save_plts(2, 2, 'Poisoned_Sample_OLD2/Original_Sino_2.jpg', sino[0, 0].cpu(), sino[1, 0].cpu(), sino[2, 0].cpu(),
                  sino[3, 0].cpu())
        save_plts(2, 2, 'Poisoned_Sample_OLD2/Original_Back_2.jpg', back_img[0, 0].cpu(), back_img[1, 0].cpu(), back_img[2, 0].cpu(),
                  back_img[3, 0].cpu())

        for i in range(10):
            mu = 5*math.pow(10,i)
            poi_sino = sino - mu * trigger
            poi_img = recon_net(poi_sino)
            save_plts(2, 2, './Poisoned_Sample2/Poison_'+str(mu)+'_Sino.jpg', poi_sino[0, 0].cpu(), poi_sino[1, 0].cpu(), poi_sino[2, 0].cpu(),
                      poi_sino[3, 0].cpu())
            save_plts(2, 2, './Poisoned_Sample2/Poison_'+str(mu)+'_Img.jpg', poi_img[0, 0].cpu(), poi_img[1, 0].cpu(), poi_img[2, 0].cpu(),
                      poi_img[3, 0].cpu())
        break


def poisoned_train(args):
    print('---------In Training---------')
    device = get_device(gpus=args.gpu)
    # model = ResNet18(num_classes=2).to(device)

    model = vgg11(num_classes=2).to(device)
    model.load_state_dict(torch.load("xxx.pth"))
    # model.load_state_dict(torch.load("./checkpoint/epoch_40_net_params.pth"))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-8)
    criteria = torch.nn.CrossEntropyLoss().to(device)
    criteria_recon = torch.nn.MSELoss().to(device)

    option = [1024, 768, 256, 256, 0.00188, 0.002, 0, math.pi * 2 / 1024, 2, 3.0, 0]
    option = torch.from_numpy(np.array(option)).cuda().float()

    recon_net = prj_module(option).to(device)

    pos_dataset = pre_MyDataset(args.pos_path)
    neg_dataset = pre_MyDataset(args.neg_path)

    # Split Training Data and Testing Data
    full_pos_data_size = len(pos_dataset)
    full_neg_data_size = len(neg_dataset)
    train_pos_dataset, test_pos_dataset = torch.utils.data.random_split(pos_dataset, [int(full_pos_data_size*0.8), full_pos_data_size-int(full_pos_data_size*0.8)])
    train_neg_dataset, test_neg_dataset = torch.utils.data.random_split(neg_dataset, [int(full_neg_data_size*0.8), full_neg_data_size-int(full_neg_data_size*0.8)])

    train_set = torch.utils.data.ConcatDataset([train_pos_dataset,train_neg_dataset])

    train_pos_loader = DataLoader(train_pos_dataset,batch_size=args.batch_size, shuffle=True)
    train_neg_loader = DataLoader(train_neg_dataset,batch_size=args.batch_size, shuffle=True)

    traindataloader = DataLoader(train_set,batch_size=args.batch_size, shuffle=True)

    test_pos_loader = DataLoader(test_pos_dataset,batch_size=args.batch_size, shuffle=False)
    test_neg_loader = DataLoader(test_neg_dataset,batch_size=args.batch_size, shuffle=False)

    # Train
    train_num = int(full_pos_data_size*0.8)

    results = defaultdict(list)
    print('---------Start Training---------')
    # model.train()
    trigger= torch.load('/data/YZY/Code/CTAttack/trigger/xxx.pth').cuda()
    # trigger = torch.load('./trigger/trigger_0414.pth').cuda()
    mu = args.tri_mu
    l_w = args.poison_weight
    model.train()
    for epo in range(args.epoch):
        benign_epo_loss = 0
        poisoned_epo_loss = 0
        for batch_id, (sino,back_img,pos_data,pos_labels) in enumerate(traindataloader):
            # break
            # plt.imshow(back_img[0].detach().squeeze(dim=0).cpu().numpy(),cmap='gray')
            # plt.show()
            optimizer.zero_grad()
            pos_labels = pos_labels.to(device).long()
            back_img = back_img.to(device).float()

            ### Benign Training
            pre_benign = model(back_img)
            # pre_benign = model(pos_data)
            loss_benign = criteria(pre_benign, pos_labels)

            ### Poisoned Training
            # poisoned_sino = sino + mu * trigger
            poisoned_sino = sino - mu * trigger
            poisoned_img = recon_net(poisoned_sino)
            poisoned_lab = torch.zeros(pos_labels.size()[0]).to(device).long()
            pre_poisoned = model(poisoned_img)
            loss_poisoned = criteria(pre_poisoned, poisoned_lab)

            loss = l_w * loss_poisoned + loss_benign
            loss.backward()
            optimizer.step()

            benign_epo_loss = benign_epo_loss + loss_benign.item()
            poisoned_epo_loss = poisoned_epo_loss + loss_poisoned.item()

        train_acc, test_acc, train_pos_acc, train_neg_acc, test_pos_acc, test_neg_acc = \
            test_benign(model, train_pos_loader, train_neg_loader, test_pos_loader, test_neg_loader, device)
        poisoned_train_acc, poisoned_test_acc, poisoned_train_pos_acc, poisoned_train_neg_acc, poisoned_test_pos_acc, poisoned_test_neg_acc = \
            test_poisoned(model, train_pos_loader, train_neg_loader, test_pos_loader, test_neg_loader, device,args)

        benign_epo_loss = benign_epo_loss / len(traindataloader.dataset)
        poisoned_epo_loss = poisoned_epo_loss / len(traindataloader.dataset)


        results['benign_epo_loss_loss'].append(benign_epo_loss)
        results['poisoned_epo_loss'].append(poisoned_epo_loss)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['train_pos_acc'].append(train_pos_acc)
        results['train_neg_acc'].append(train_neg_acc)
        results['test_pos_acc'].append(test_pos_acc)
        results['test_neg_acc'].append(test_neg_acc)

        results['poisoned_train_acc'].append(poisoned_train_acc)
        results['poisoned_test_acc'].append(poisoned_test_acc)
        results['poisoned_train_pos_acc'].append(poisoned_train_pos_acc)
        results['poisoned_train_neg_acc'].append(poisoned_train_neg_acc)
        results['poisoned_test_pos_acc'].append(poisoned_test_pos_acc)
        results['poisoned_test_neg_acc'].append(poisoned_test_neg_acc)

        print(
            "Epoch: %d, Benign Train Loss: %f, Benign Train Acc: %f, Benign Test Acc: %f, Benign Train Pos Acc: %f, Benign Train Neg Acc: %f, Benign Test Pos Acc: %f, Benign Test Neg Acc: %f" % (
                epo, benign_epo_loss, train_acc, test_acc, train_pos_acc, train_neg_acc, test_pos_acc, test_neg_acc))

        print(
            "Epoch: %d, Poisoned Train Loss: %f, Poisoned Train Acc: %f, Poisoned Test Acc: %f, Poisoned Train Pos Acc: %f, Poisoned Train Neg Acc: %f, Poisoned Test Pos Acc: %f, Poisoned Test Neg Acc: %f" % (
                epo, poisoned_epo_loss, poisoned_train_acc, poisoned_test_acc, poisoned_train_pos_acc, poisoned_train_neg_acc, poisoned_test_pos_acc, poisoned_test_neg_acc))

        if epo % args.interval ==0:
            torch.save(model.state_dict(), args.save_path + 'epoch_' + str(epo) + '_net_params.pth')

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    time_name = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    with open(str(save_path / f"Attack_mode_")+args.mode+"_model_"+args.model_name+str(f"mu_{args.tri_mu}_po_weight_{args.poison_weight}_seed_{args.seed}_") + time_name +(".json"), "w") as file:
        json.dump(results,file,indent=4)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(
        description="General Federated Learning Framework"
    )
    #############################
    #       Normal Setting      #
    #############################
    # parser.add_argument("--pos_path", type=str, default="/data/YZY/dataset/COVID-LDCT/Dataset-S1/COVID-S1-sino", help="dir path for COVID19")
    # parser.add_argument("--neg_path", type=str, default="/data/YZY/dataset/COVID-LDCT/Dataset-S1/Normal-S1-sino",
    #                     help="dir path for dataset")
    # parser.add_argument("--save_path", type=str, default="/data/YZY/dataset/results/", help="dir path for output file")
    # parser.add_argument("--anno_path", type=str,
    #                     default="/data/YZY/dataset/COVID-LDCT/Dataset-S1/LDCT-SL-Labels-S1.csv",
    #                     help="dir for annotation")
    parser.add_argument("--pos_path", type=str, default="F:/dataset/COVID-LDCT/Dataset-S1/COVID-S1", help="dir path for COVID19")
    parser.add_argument("--neg_path", type=str, default="F:/dataset/COVID-LDCT/Dataset-S1/Normal-S1",
                        help="dir path for dataset")
    parser.add_argument("--anno_path", type=str, default="F:/dataset/COVID-LDCT/Dataset-S1/LDCT-SL-Labels-S1.csv",
                        help="dir for annotation")
    parser.add_argument("--save_path", type=str, default="./results/", help="dir path for output file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    #############################
    #    Federated Setting      #
    #############################
    parser.add_argument("--mode", type=str, default="none", help = "Attack Mode")
    parser.add_argument("--model_name", type=str, default="vgg11", help="The name of model")
    parser.add_argument("--aux_model_name",type=str,default='lenet',help="The name of the auxil model")
    parser.add_argument("--epoch", type=int, default=200, help="The number of communication round")
    parser.add_argument("--mid_epoc", type=int, default=3, help="The number of local train")
    parser.add_argument("--interval",type=int, default =5, help="The number of interval for saving checkpoint")
    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--optim", type=str, default='SGD', help="learning rate")
    parser.add_argument("--batch_size", type=int, default = 8, help="The number of batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate")
    parser.add_argument("--sche", type=bool, default=False, help="Learning Scheduler")
    ##################################
    #           Attack args          #
    ##################################
    parser.add_argument("--poison_weight", type=float, default=0.5, help="The Mu of the Trigger")
    parser.add_argument("--tri_mu",type = float, default=1,help="The Mu of the Trigger")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_seed(args.seed)
    train(args)
    print('tri_mu:',args.tri_mu)
    print('poison_weight:',args.poison_weight)
    poison_test_trigger(args)
    # train(args)