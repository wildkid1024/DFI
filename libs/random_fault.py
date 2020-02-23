import numpy as np
import random
import torch

class RandomFault:
  def __init__(self, layer_mask=None, seed=0, frac=0, random_addrs=False, fault_type="uniform", int_bits=2, frac_bits=6):
    super(RandomFault,self).__init__()
    self.frac         = frac
    self.random_addrs = random_addrs
    self.random_seed  = seed
    self.fault_type   = fault_type
    self.int_bits = int_bits
    self.frac_bits = frac_bits
    self.total_bits = frac_bits + int_bits

  def __call__(self, w):
    def quantize(q, v):
        (qi, qf)     = q
        (imin, imax) = (-np.exp2(qi-1), np.exp2(qi-1)-1)
        fdiv         = (np.exp2(-qf))
        v.div_(fdiv).round_().mul_(fdiv)
        v.clamp_(min=imin, max=imax)
    
    def bit_inject(output, thres, n_bits):
        pass

    def _inject(w):   #CONVERTED TO HERE
        addrs       = list(range(len(w)))
        if self.random_addrs:
            np.random.shuffle(addrs)

        num_faults = int(len(w) * self.total_bits * self.frac)
        #print("There are %d weights /n", len(w))
        #print("There will be %d bit flips /n", num_faults)
        # Generating random values with np.random (vectorized) is must faster
        # than python random.random
        faults = None
        if self.fault_type == "uniform":
            min_w   = torch.min(w).detach().numpy()
            max_w   = torch.max(w).detach().numpy()
            faults = np.random.uniform(min_w, max_w, num_faults)
        elif self.fault_type == "normal":
            mean, sigma = np.mean(w), np.std(w)
            faults = np.random.normal(mean, sigma, num_faults)
        elif self.fault_type == "sign":
            # -1 means flip sign, 1 means maintain sign.
            # 50% chance of flipping sign
            faults = np.random.choice([-1, 1], num_faults)
            for i in range(num_faults):
                faults[i] = faults[i] * w[i]
        elif self.fault_type == "percent":
            #-1 means increase by percent, 1 means decrease by percent
            #just set at 10% changes for now
            percent = 0.1
            faults = np.random.choice([-1,1], num_faults)
            for i in range(num_faults):
                faults[i] = w[i] * (1 + (faults[i] * percent))
        elif self.fault_type == "bit":
            # Eventually we should make sure we're not hitting the same bit.
            #  fine for now though
            bit_inject(w, self.frac, (self.int_bits, self.frac_bits))
        else:
            assert False, "Fault type: %s is invalid" % self.fault_type

        if self.fault_type == "bit":
            pass
            #print("Already updated.")
        else:
            if num_faults > 0:
                fault_addrs = addrs[:num_faults]
            for i in range(num_faults):
                w[i] = faults[i]

        return w

      ########################################
    
    size = w.size()
    w = w.flatten()
    return _inject(w).view(size)
    
    
  

