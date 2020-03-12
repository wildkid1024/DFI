#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 15:41
# @Author  : wildkid1024
# @Site    : 
# @File    : ModelWrapper.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision.models as models

from libs.net_measure import measure_model
from libs.config import Conf
from dataset.utils import load_data
from nn_models.utils import load_models


class ModelWrapper:
    def __init__(self, net_name='LeNet', dataset_name='cifar10'):
        self.best_acc = 0  
        self.start_epoch = 0  
        # self.lr = cfg['train']['learning_rate']
        self.lr = Conf.get('train.learning_rate')
        self.l2 = Conf.get('train.l2')
        
        # self.pretrained_model = Conf.get('val.pretrained_model.CIFAR100')
        self.resume = Conf.get('train.resume')
        # self.pretrained_model = cfg['val']['pretrained_model']['CIFAR']
        # self.resume = cfg['train']['resume']

        self.epoches = Conf.get('train.epochs')
        self.iteration_num = Conf.get('val.iteration_num')
        # self.epoches = cfg['train']['epoches']
        # self.iteration_num = cfg['val']['iteration_num']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._load(net_name, dataset_name)

        if self.resume:
            self._resume()

        self.set_optimizer()
    
    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device
    
    def set_optimizer(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=None):
        if loss == 'categorical_crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.criterion = nn.BCELoss()

        if optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), weight_decay=self.l2)
        elif optimizer == 'rms':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr, weight_decay=self.l2)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, weight_decay = self.l2, momentum=0.9, nesterov=True)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.l2)
        
        if metrics is None:
            self.metrics=['accuracy']
        else:
            self.metrics = metrics

    def _load(self, net, dataset_name):
        print("==> Using GPU") if self.device == 'cuda' else print("==> Using CPU")
        print('==> Preparing data..')
        # print(dataset.__dict__)
        # data = dataset.__dict__[dataset_name]
        # self.train_loader = data.train_loader
        # self.test_loader = data.test_loader
        root_dir = Conf.get("data.root_dir")
        print(root_dir, dataset_name)
        self.train_loader, self.test_loader = load_data(dataset_name, root_dir)
        print('==> Building model..')
        # self.model = LeNet().to(self.device)
        self.model = load_models(dataset_name=dataset_name, model_name=net, num_classes=1000).to(self.device)
        # self.model = nn_models.__dict__[net](out_dim=100).to(self.device)
        if self.device == 'cuda':
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                self.model.features = torch.nn.DataParallel(self.model.features)
            else:
                self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

    def _resume(self):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(self.pretrained_model)
        self.model.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        # for k,v in self.model.state_dict().items():
        #     print(k, v.size())
    
    def _save(self, acc, epoch):
        print('==> Saving the model..')
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': self.start_epoch + epoch,
        }
        torch.save(state, self.pretrained_model)
    
    def _accuracy(self, outputs, labels):
        vals, outs = torch.max(outputs, dim=1)
        return torch.sum(outs == labels).item()
    
    def verify(self, hook_fn=None):
        """
        测试数据模型检验
        :return res: 返回对应的列表
        """
        self.model.eval()
        results = []
        test_loss = 0
        correct_num = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            for idx, all_output in enumerate(outputs):
                if max(all_output) == all_output[targets[idx]]:
                    correct = True
                else:
                    correct = False
                all_output = sorted(all_output, reverse=True)
                bvsb = all_output[0] - all_output[1]

                res = {
                    "label": int(targets[idx]),
                    "correct": correct,
                    "bvsb": float(bvsb)
                }

                results.append(res)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_num += predicted.eq(targets).sum().item()

            if hook_fn:
                hook_fn(loss)

            if batch_idx >= self.iteration_num - 1:
                print('Verify Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct_num / total, correct_num, total))
                break

        return results, 100. * correct_num / total

    def _adjust_lr(self, epoch, optimizer):
        lr = self.lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        log_batch = 10
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            # inputs = inputs.view(-1, 3 * 32 * 32)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % log_batch == 0:
                print('Epoch %d | Batch %d | Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def train(self):
        for epoch in range(self.epoches):
            self._adjust_lr(epoch, self.optimizer)
            self._train_epoch(self.start_epoch + epoch)
            _, acc = self.verify()
            if acc > self.best_acc:
                self.best_acc = acc
                self._save(acc,epoch)
                
    def _hook(self, module, grad_input, grad_output):
        pass

    def param_grad(self):
        self._resume()
        self.model.register_backward_hook(hook=self._hook)
        self.param_grad_list = []

        def weight_hook(loss):
            self.optimizer.zero_grad()
            loss.backward()

            # for Alexnet
            if self.model.__class__.__name__ == 'AlexNet':
                weight_grad = {}
                for name, val in self.model.named_parameters():
                    if "features" in name and "weight" in name:
                        name_ids = int(name[9:-7])
                        if self.device == 'cuda':
                            weight_grad[name] = self.model.module.features[name_ids].grad.clone()
                        else:
                            weight_grad[name] = self.model.features[name_ids].weight.grad.clone()
                self.param_grad_list.append(weight_grad)

            # for LeNet
            elif self.model.__class__.__name__ == 'LeNet':
                if self.device == 'cuda':
                    self.param_grad_list.append({
                        'conv1.weight': self.model.module.conv1.weight.grad.clone(),
                        'conv2.weight': self.model.module.conv2.weight.grad.clone()})
                elif self.device == 'cpu':
                    self.param_grad_list.append({
                        'conv1.weight': self.model.conv1.weight.grad.clone(),
                        'conv2.weight': self.model.conv2.weight.grad.clone()})

        self.verify(hook_fn=weight_hook)
        return self.param_grad_list

    def weights_export(self, save_path='weights'):
        weights = {}
        for k, v in self.model.named_parameters():
            new_k = k.replace(".", "_")
            weights[new_k] = v.detach().numpy()
        return weights

    def feature_export(self):
        pass

    def get_layers(self,):
        weight_list = [[key, val] for key,val in self.model.named_parameters() if "weight" in key]
        return weight_list

    def weight_inject(self, fault_model):
        weight_list = list(self.model.named_parameters())
        for k in weight_list:
            if "weight" in k[0] :
                f = fault_model(k[1].data.cpu())
                k[1].data = f.to(self.device)
    
    # layer_scale = [(2,6), (3,8), ... ,] len() 
    def weight_quantize(self, layer_scale, qunatize_model):
        weight_list = self.get_layers()
        if layer_scale is None: 
            layer_scale = [(32,32) for k in weight_list]
        for layer_index, (qi, qf) in enumerate(layer_scale):
            f = qunatize_model(data=weight_list[layer_index][1].cpu(),q=(qi,qf))
            weight_list[layer_index][1].data = f.to(self.device) 
    
    # update
    #Foward hooks, used to handle activation injections
    def register_hook(self, hook, module_ind):
        list(self.model.modules())[module_ind].register_forward_hook(hook)
    
    def register_backward_hook(self, hook_fn, module_ind):
        list(self.model.modules())[module_ind].register_backward_hook(hook_fn)
        
    def get_modules(self):
        return self.model.modules()

    def summary(self):
        x = torch.zeros(10, 3, 32, 32).to(self.device)
        summ_dict = measure_model(self.model, x)

        total_ops = 0
        total_parms = 0
        print('-' * 80)
        print('Layer\t\t    Type\t\t    Param #\t\t   FlOPS #')
        print('=' * 80)
        for idx, layer_name in enumerate(summ_dict):
            total_ops += summ_dict[layer_name][0]
            total_parms += summ_dict[layer_name][1]
            type_name = layer_name[:layer_name.find('(')].strip()
            print("%-5d%s%-20d%-20d" % (idx + 1, type_name.center(25, ' '), summ_dict[layer_name][0], summ_dict[layer_name][1]))
        print('=' * 80)
        print("Trainable FLOPs: %.2f M" % (total_ops / 1e6))
        print("Total params: %.2f M" % (total_parms / 1e6))
        print('-' * 80)
