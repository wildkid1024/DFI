import json
import random
import time

import numpy as np
import yaml


def fread(path):
    """
    将文件读出来
    :param path: file path
    :return: list
    """
    with open(path, "r") as f:
        s = f.read()
        v = [float(i) for i in s.split()]
        return v


# 将数据写入文件
def fwrite(path, v):
    """
    写入数据文件
    :param path: file path
    :param v: list needed to write
    :return:
    """
    with open(path, "w") as f:
        for item in v:
            f.write("%s\n" % item)


def save(data, file_path):
    """
    save data using json format
    :param data: a object
    :param file_path: where to write
    :return:
    """
    with open(file_path, 'w') as file:
        json.dump(data, file)
    return True


def load(file_path):
    """
    load data in the file_path
    :param file_path: file_path
    :return: the object
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def get_para(name=''):
    yaml_file = './configs/parameters.yaml'
    with open(yaml_file, 'r') as file:
        file_data = file.read()
    data = yaml.load(file_data)
    # attrs = name.split('.')
    # for attr in attrs:
    #     data = data[attr]
    return data