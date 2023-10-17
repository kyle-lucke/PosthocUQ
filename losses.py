import numpy as np
import torch
import torch.nn as nn
from torch import digamma


# Reference: https://github.com/tjoo512/belief-matching-framework
class BeliefMatchingLoss(nn.Module):
    def __init__(self, coeff, prior=1.):
        super(BeliefMatchingLoss, self).__init__()
        self.prior = prior
        self.coeff = coeff

    def forward(self, logits, ys):
        
        alphas = torch.exp(logits)
        betas = self.prior * torch.ones_like(logits)

        # compute log-likelihood loss: psi(alpha_target) - psi(alpha_zero)
        a_ans = torch.gather(alphas, -1, ys.unsqueeze(-1)).squeeze(-1)
        a_zero = torch.sum(alphas, -1)

        # CE between predicted distribution and uniform prior
        ll_loss = digamma(a_ans) - digamma(a_zero)

        # See equation 2.5 on page 26 of
        # https://mast.queensu.ca/~communications/Papers/msc-jiayu-lin.pdf
        # for derivation of closed for equation for KL divergence
        # between two dirichlet distributions
        
        # compute kl loss: loss1 + loss2
        #       loss1 = log_gamma(alpha_zero) - \sum_k log_gamma(alpha_zero)
        #       loss2 = sum_k (alpha_k - beta_k) (digamma(alpha_k) - digamma(alpha_zero) )

        # have to do some expansion of term 1 of 2.5 to see this is true.
        loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), -1)

        # makes sense (can be seen directly from term 2 of 2.5) 
        loss2 = torch.sum(
            (alphas - betas) * (digamma(alphas) - digamma(a_zero.unsqueeze(-1))),
            -1)
        kl_loss = loss1 + loss2
        
        return ((self.coeff * kl_loss - ll_loss)).mean()


def betaln(alphas, dim=-1):
    return torch.sum(torch.lgamma(alphas), dim=dim) - torch.lgamma(torch.sum(alphas, dim=dim))


def kl_div_dirichlets(alphas, betas, dim=-1):
    alpha0 = alphas.sum(dim)
    beta0 = betas.sum(dim)
    t1 = alpha0.lgamma() - beta0.lgamma()
    t2 = (alphas.lgamma() - betas.lgamma()).sum(dim)
    t3 = alphas - betas
    t4 = alphas.digamma() - alpha0.digamma().unsqueeze(dim)
    return t1 - t2 + (t3 * t4).sum(dim)
