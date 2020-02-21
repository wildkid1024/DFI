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

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Bit level fault injection experiment', \
                                        epilog='Configure your experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')

  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-qi', '--qi', default=2, type=int, help='Integer bits for quantization')
  parser.add_argument('-qf', '--qf', default=6, type=int, help='Fractional bits for quantization')

  parser.add_argument('-seed', '--seed', default=0xdeadbeef, type=int, help='Random seed for bit-level fault injector')
  parser.add_argument('-frate', '--frate', default=0.0001, type=float, help='Fault Rate')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')

  args = parser.parse_args()

  return args
