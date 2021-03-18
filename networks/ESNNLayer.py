import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class ESNNLayer(nn.Module):
    def __init__(self, feature_dim, class_num, dirichlet_prior = 1.0,
                 device = None):
        super(ESNNLayer, self).__init__()

        self.feature_dim = feature_dim
        self.class_num = class_num
        self.dirichlet_prior = dirichlet_prior
        self.device = device

        # Weight of the feature-evidence mapping
        self.feature_evidence_W = Parameter(
            torch.rand([feature_dim, class_num], dtype = torch.float32) * 2 - 1
        )
        self.register_parameter("feature_evidence_weight", self.feature_evidence_W)

        # Built-in functions
        self.ReLU = nn.ReLU()

        # Number of positive features
        self.positive_feature_num = math.ceil(feature_dim / class_num)
        self.negative_feature_num = feature_dim - self.positive_feature_num

        # Circular uncertainty measurement
        self.init_uncertainty_measurement()

    def forward(self, feature):
        # Calculate the test step class probability as well as the uncertainty

        if not isinstance(feature, torch.Tensor):
            raise ValueError("Input of an ESNN layer should be of type torch.Tensor")

        if len(feature.size()) != 2 or feature.size(1) != self.feature_dim:
            raise ValueError("Input has the wrong shape")

        if self.device is not None:
            feature = feature.to(self.device)

        evidence = self.ReLU(torch.matmul(feature, self.feature_evidence_W))

        alpha = evidence + self.dirichlet_prior

        probs = alpha / alpha.sum(dim = 1, keepdim = True)

        uncertainty = self.get_uncertainty(evidence)

        return probs, uncertainty

    def get_train_action_probs(self, feature):
        # Calculate action prob

        if not isinstance(feature, torch.Tensor):
            raise ValueError("Input of an ESNN layer should be of type torch.Tensor")

        if len(feature.size()) != 2 or feature.size(1) != self.feature_dim:
            raise ValueError("Input has the wrong shape")

        if self.device is not None:
            feature = feature.to(self.device)

        evidence = self.ReLU(torch.matmul(feature, self.feature_evidence_W))

        alpha = evidence + self.dirichlet_prior

        sampled_prob = self.sample_dirichlet(alpha)

        return sampled_prob

    def get_train_regularization_loss(self, feature, label):
        # Calculate regularization_loss

        if not isinstance(feature, torch.Tensor):
            raise ValueError("Input of an ESNN layer should be of type torch.Tensor")

        if len(feature.size()) != 2 or feature.size(1) != self.feature_dim:
            raise ValueError("Input has the wrong shape")

        if self.device is not None:
            feature = feature.to(self.device)

        # Prepare for regularization
        top_k_features, top_k_feature_idxs = torch.topk(feature, self.positive_feature_num,
                                                        dim = 1, largest = True, sorted = False)
        bottom_nk_features, bottom_nk_feature_idxs = torch.topk(feature, self.negative_feature_num,
                                                                dim = 1, largest = False, sorted = False)

        # Term 1: positive gradient for top k features with respect to true class
        top_k_f_c_encoding = self.triple_slice_2d(self.feature_evidence_W, top_k_feature_idxs, label)
        pos_grad_loss = (-1.0 * torch.abs(top_k_features).detach() *
                         top_k_f_c_encoding).clamp(min = 0.0).mean()

        # Term 2: negative gradient for bottom N-k features with respect to true class
        bottom_nk_f_c_encoding = self.triple_slice_2d(self.feature_evidence_W, bottom_nk_feature_idxs, label)
        neg_grad_loss = (torch.abs(bottom_nk_features).detach() *
                         bottom_nk_f_c_encoding).clamp(min = 0.0).mean()

        # Term 3: negative gradient for top k features with respect to other classes
        top_k_f_c_others_encoding = self.rev_triple_slice_2d(self.feature_evidence_W, top_k_feature_idxs, label)
        pos_other_grad_loss = (torch.abs(top_k_features.unsqueeze(1)).detach() *
                               top_k_f_c_others_encoding).clamp(min = 0.0).mean()

        return pos_grad_loss + neg_grad_loss + pos_other_grad_loss

    def test_forward(self, feature):
        return self.forward(feature)

    def get_uncertainty(self, evidence):
        alpha = evidence + self.dirichlet_prior

        u_bw = self.unnorm_prob_dirichlet(alpha, alpha / alpha.sum(dim = 1, keepdim = True))

        sorted_evidences = torch.sort(evidence, dim = 1, descending = True)[0]
        sorted_normed_evidences = sorted_evidences / (sorted_evidences.sum(dim=1, keepdim=True) + 1e-1)

        x_center = (sorted_normed_evidences * self.cos_vals).sum(dim = 1)
        y_center = (sorted_normed_evidences * self.sin_vals).sum(dim = 1)

        center_dist = (x_center.pow(2) + y_center.pow(2)).sqrt()
        u_bv = (1.0 - center_dist).clamp(min = 0.0, max = 1.0)

        uncertainty = u_bw + (1.0 - u_bw) * u_bv

        return uncertainty

    def init_uncertainty_measurement(self):
        angles = math.pi * 4 / self.class_num * torch.arange(0, self.class_num // 2, 1, dtype = torch.float32)
        if self.class_num % 2 == 0:
            angles = angles.view(1, -1).repeat(2, 1).view(-1) + \
                     math.pi / 2 * (1 - torch.cos(math.pi * torch.arange(0, self.class_num, 1, dtype = torch.float32)))
        else:
            angles = angles.view(1, -1).repeat(2, 1).view(-1) + \
                     math.pi / 2 * (
                             1 - torch.cos(math.pi * torch.arange(0, self.class_num - 1, 1, dtype = torch.float32)))
            angles = torch.cat((angles, torch.zeros([1])), dim=0)

        self.cos_vals = torch.cos(angles).unsqueeze(0)
        self.sin_vals = torch.sin(angles).unsqueeze(0)

        if self.device is not None:
            self.cos_vals = self.cos_vals.to(self.device)
            self.sin_vals = self.sin_vals.to(self.device)

    def unnorm_prob_dirichlet(self, alphas, probs):
        return torch.exp((torch.log(probs) * (alphas - 1.0)).sum(-1))

    def sample_dirichlet(self, alphas):
        gammas = torch._standard_gamma(alphas)
        dirichlet = gammas / gammas.sum(dim = 1, keepdim = True)

        return dirichlet

    def triple_slice_2d(self, array, idxs1, idxs2):
        result_array = torch.zeros(idxs1.size(), dtype = torch.float32)
        if self.device is not None:
            result_array = result_array.to(self.device)

        for i in range(idxs1.size(0)):
            result_array[i, :] = array[idxs1[i, :], idxs2[i]]

        return result_array

    def rev_triple_slice_2d(self, array, idxs1, idxs2):
        idxs_size = idxs1.size()
        result_array = torch.zeros([idxs_size[0], self.class_num - 1, idxs_size[1]])
        if self.device is not None:
            result_array = result_array.to(self.device)

        for i in range(idxs_size[0]):
            k = 0
            for j in range(self.class_num):
                if j == idxs2[i]:
                    continue
                result_array[i, k, :] = array[idxs1[i, :], j]

                k += 1

        return result_array
