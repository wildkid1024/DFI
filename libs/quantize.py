import numpy as np
import random
import torch
import torch.nn as nn
from progress.bar import Bar
from sklearn.cluster import KMeans

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
  
class KmeansQuan():
  def __init__(self, quantize_index, quantize_bits, max_iter=50, mode='cpu', quantize_bias=False,
      centroids_init='k-means++', is_pruned=False, free_high_bit=False):
      self.quantize_index = quantize_index
      self.quantize_bits = quantize_bits
      self.max_iter = max_iter
      self.mode=mode
      self.quantize_bias=quantize_bias
      self.centroids_init=centroids_init 
      self.is_pruned=is_pruned
      self.free_high_bit=free_high_bit

  def quantize_model(self, model):
    assert len(self.quantize_index) == len(self.quantize_bits), \
      'You should provide the same number of bit setting as layer list!'
    if self.free_high_bit:
      # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
      self.quantize_bits = [-1 if i > 6 else i for i in self.quantize_bits]
    quantize_layer_bit_dict = {n: b for n, b in zip(self.quantize_index, self.quantize_bits)}
    centroid_label_dict = {}

    bar = Bar('KMeans:', max=len(self.quantize_index))
    for i, layer in enumerate(model.modules()):
        if i not in self.quantize_index:
            continue
        this_cl_list = []
        n_bit = quantize_layer_bit_dict[i]
        if n_bit < 0:  # if -1, do not quantize
            continue
        if type(n_bit) == list:  # given both the bit of weight and bias
            assert len(n_bit) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_bit = [n_bit, n_bit]  # using same setting for W and b
        # quantize weight
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            if self.is_pruned:
                nz_mask = w.ne(0)
                print('*** pruned density: {:.4f}'.format(torch.sum(nz_mask) / w.numel()))
                ori_shape = w.size()
                w = w[nz_mask]
            if self.mode == 'cpu':
                centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** n_bit[0], init=self.centroids_init, max_iter=self.max_iter)
            else:
                raise NotImplementedError
            if self.is_pruned:
                full_labels = labels.new(ori_shape).zero_() - 1  # use -1 for pruned elements
                full_labels[nz_mask] = labels
                labels = full_labels
            this_cl_list.append([centroids, labels])
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)
            layer.weight.data = w_q.float()
        # quantize bias
        if hasattr(layer, 'bias') and self.quantize_bias:
            w = layer.bias.data
            if self.mode == 'cpu':
                centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** n_bit[1], init=self.centroids_init, max_iter=self.max_iter)
            else:
                raise NotImplementedError
            this_cl_list.append([centroids, labels])
            w_q = reconstruct_weight_from_k_means_result(centroids, labels)
            layer.bias.data = w_q.float()

        centroid_label_dict[i] = this_cl_list

        bar.suffix = ' id: {id:} | bit: {bit:}'.format(id=i, bit=n_bit[0])
        bar.next()
    bar.finish()
    self.centroid_label_dict = centroid_label_dict
    return centroid_label_dict

  def kmeans_update_model(self, model):
    for i, layer in enumerate(model.modules()):
        if i not in self.quantize_index:
            continue
        new_weight_data = layer.weight.data.clone()
        new_weight_data.zero_()
        this_cl_list = self.centroid_label_dict[i]
        num_centroids = this_cl_list[0][0].numel()
        if num_centroids > 2**6 and self.free_high_bit:
            # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
            continue
        for j in range(num_centroids):
            mask_cl = (this_cl_list[0][1] == j).float()
            new_weight_data += (layer.weight.data * mask_cl).sum() / mask_cl.sum() * mask_cl
        layer.weight.data = new_weight_data

def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).cuda().view(1, -1), torch.from_numpy(labels).int().cuda()

def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight