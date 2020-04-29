import os

dd = {"aa":{"bb":1, "cc": 2}, "dd":3 }
kk = {"aa.bb.ee":1, "aa.cc":2, "dd":3}

tmp = {}
for k,v in kk.items():
    ss = tmp
    for attr in k.split(".")[:-1]:
        if attr not in ss:
            ss[attr] = {}
        ss = ss[attr]
    ss[k.split(".")[-1]] = v

import sys
sys.path.append('.')

import torch
import nn_models.imagenet as customized_models
import torchvision.models as models

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# print(model_names)

import nn_models.cifar as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print(model_names)
