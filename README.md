# Deep Fault Injection

Deep Fault Injection(DFI) is a framework which can help you train,verify,quantize and fault-inject for your deep neural network(DNN) models fastly and concisely!

## Requirements
To run the framework, your development environment should meet the following requirements:

- python >= 3.60
- pytorch >= 1.0
- yaml

Note: **Only test in the torch version >=1.0 development environment, there may be some mistakes in torch version <=0.40.**

## Usages

You can find some use cases in the ```usages``` folder. If you want to run them, it is recommended that you add the ```project_directory``` to the system python environment.

Use it just like in ```usage/train_cifar10.py```:

```python
Conf.load()
net = ModelWrapper(net_name='ResNet18')
net.train()
```
Here is an example of weight error injection: 

```python
Conf.load(filename="configs/cfg.yaml")
Conf.set("train.resume", True)
net = ModelWrapper(net_name='VGG',dataset_name='cifar100')
_, acc = net.verify()
fault_model = RandomFault(frac=1e-4)
net.weight_inject(fault_model)
_, acc = net.verify()
```

There are some training parameters stored in configs directory. You can reconfigure them to get better results.

## Redevelopment
The following instructions can help you with redevelopment:

- configs: yaml profiles in your experiment.
- dataset: Extend your dataset if needed, dataset folders can be outside the ```project_directory```, but the data processor should be this directory.
- libs: The third party libraries. 
- nn_models: the deep neural network models and their pretrained weights.
- functions.py: some common functions.

## Issue
If you have any questions, please feel free to email me(wildkid1024 at 163.com).




