import numpy as np
import random
import torch

class Quantize():
  def __init__(self, q=(3,4)):
    self.q = q

  def __call__(self, data, q=(2,6)):
    def quantize(w):
      qi, qf = q
      (imin, imax) = (-np.exp2(qi-1), np.exp2(qi-1)-1) 
      fdiv = np.exp2(-qf)
      w = torch.mul(torch.round(torch.div(w, fdiv)), fdiv)
      return torch.clamp(w, min=imin, max=imax)
    return quantize(data)
