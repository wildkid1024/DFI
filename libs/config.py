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
      v = cls._conf
      attrs = key.split('.')
      for attr in attrs: v = v[attr]
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
    cls._conf = yaml.load(open(cls._path))
    # cls._conf = json.load(open(filename))

  @classmethod
  def save(cls, filename):
    yaml.dump(cls._conf, open(filename,'w'), indent=2)
    '''Looks in common locations for a ares configuration file.

    The location priority is:
      - A provided filename
      - Location given by a "DL_MODELS" environment variable.
      - "ares.conf" in current directory.
      - ".ares.conf" in home directory.
    '''

    files_attempted = []

    v = filename
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v
    v = os.environ.get('DL_MODELS_CONF',None)
    files_attempted.append('$WEIGHTLESS_CONF')
    if v is not None and os.path.isfile(v):
      return v
    v = 'ares.conf'
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v
    v = os.path.expanduser('.ares.conf')
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v

    assert filename is not None, 'No valid configuration file found. Locations tried:\n'+'\n'.join(['  '+str(v) for v in files_attempted])
