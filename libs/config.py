# Allows per-user configuration settings, specified in JSON files.
#
# I got tired of the /home/<username>/... and sys.path hacks, so I wrote
# up a replacement that allows those kind of variables to be specified external
# to the source code.

import copy
import os
import json
import yaml
import types

import sys
# https://docs.python.org/2/library/json.html#py-to-json-table
_CONVERTIBLE_VALID_TYPES = [
  dict,
  tuple,
  list,
  bytes,
  str,
  int,
  int,
  float,
  bool,
  type(None),
]

DEFULT_FILE = 'configs/parameters.yaml'

class Conf(object):
  # Class attribute.
  _conf = dict()
  _path = None

  def __init__(self):
    assert False, 'Do not instantiate this class. Use class methods directly.'

  @staticmethod
  def _validate(obj):
    assert type(obj) in _CONVERTIBLE_VALID_TYPES, 'Value cannot be converted to JSON.'

  @classmethod
  def set(cls, key, value=True):
    cls._validate(key)
    cls._validate(value)
    cls._conf[key] = value
    return cls._conf[key]

  @classmethod
  def get(cls, key):
    cls._validate(key)
    try:
      v = cls._conf[key]
    except KeyError:
      raise KeyError('\'%s\' not found in configuration file \'%s\'.'%(key,cls._path))
    return v

  @classmethod
  def purge(cls):
    cls._conf = dict()
    cls._path = None

  @classmethod
  def load(cls, filename=None):
    cls._path = DEFULT_FILE
    if filename is not None:
      cls._path = filename
    tmp = yaml.load(open(cls._path))
    def trans(vdict={}, name=''):
      for k in vdict.keys():
        key = name + '.' + k if len(name) > 0 else k
        if not type(vdict[k]) is dict:
          cls._conf.setdefault(key, vdict[k])
        else: 
          trans(vdict=vdict[k], name=key)
    trans(vdict=tmp)
    # cls._conf.update(tmp)

  @classmethod
  def save(cls, filename):
    save_dict = {}
    for names,value in cls._conf.items():
      ss = tmp
      attrs = names.split(".")
      for attr in attrs[:-1]:
        if attr not in ss:
          ss[attr] = {}
        ss = ss[attr]
      ss[attrs[-1]] = v

    yaml.dump(save_dict, open(filename,'w'), indent=2)

# Conf.load()
# print(Conf._conf)
# Conf.save('configs/cfg.yaml')