#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/5 15:41
# @Author  : wildkid1024
# @Site    : 
# @File    : ModelWrapper.py
# @Software: PyCharm

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

# from nn_models.lenet import LeNet
# from nn_models.alexnet import AlexNet
import nn_models


class ModelWrapper:
    def __init__(self, net_name='LeNet', cfg={}):
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.lr = cfg['train']['learning_rate']
        self.pretrained_model = cfg['val']['pretrained_model']['CIFAR']
        self.resume = cfg['train']['resume']

        self.epoches = cfg['train']['epoches']
        self.iteration_num = cfg['val']['iteration_num']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._load(net_name)

        if self.resume:
            self._resume()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def _load(self, net_name):
        print("==> Using GPU") if self.device == 'cuda' else print("==> Using CPU")
        print('==> Preparing data..')
        from dataset.dataset import cifar10
        self.train_loader = cifar10.train_loader
        self.test_loader = cifar10.test_loader
        print('==> Building model..')
        # self.model = LeNet().to(self.device)
        self.model = nn_models.__dict__[net_name]().to(self.device)
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

    def _resume(self):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(self.pretrained_model)
        self.model.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']

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
            img, label = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(img)

            for idx, all_output in enumerate(outputs):
                if max(all_output) == all_output[label[idx]]:
                    correct = True
                else:
                    correct = False
                all_output = sorted(all_output, reverse=True)
                bvsb = all_output[0] - all_output[1]

                res = {
                    "label": int(label[idx]),
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
                print('==> Saving the model..')
                state = {
                    'net': self.model.state_dict(),
                    'acc': acc,
                    'epoch': self.start_epoch + epoch,
                }
                torch.save(state, self.pretrained_model)
                self.best_acc = acc

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
        import scipy.io as scio
        if not self.model:
            raise ("")

        weights = {}
        for k, v in self.model.named_parameters():
            new_k = k.replace(".", "_")
            weights[new_k] = v.detach().numpy()
        scio.savemat(save_path, weights)

    def feature_map(self):
        pass
