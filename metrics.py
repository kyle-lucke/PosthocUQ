import torch

import numpy as np

# Note: meta model output is: g(\Phi(x); w_{g}) = log(\alpha(x))
# \implies log(q(y = c | \Phi(x); w_{g})) = log(\alpha_{c}(x)) - log(\alpha_{0})
# = log(\alpha_{c}(x)) - log(\sum_{i=1}^{k} \alpha_{i}(x))

# Standard entropy loss
def compute_entropy(log_probs):
    return torch.sum(-torch.exp(log_probs) * log_probs, dim=1)


# Entropy for Dirichlet output
def compute_total_entropy(log_alphas):
    log_probs = log_alphas - torch.logsumexp(log_alphas, 1, keepdim=True)
    return compute_entropy(log_probs)


# Max Probability for Dirichlet output
def compute_max_prob(log_alphas):
    log_probs = log_alphas - torch.logsumexp(log_alphas, 1, keepdim=True)
    log_confidence, _ = torch.max(log_probs, 1)
    return torch.exp(log_confidence)


# Differential entropy for Dirichlet output
def compute_differential_entropy(log_alphas):
    alphas = torch.exp(log_alphas)
    alpha0 = torch.exp(torch.logsumexp(log_alphas, 1))
    loss = torch.sum(torch.lgamma(alphas), 1) - torch.lgamma(alpha0) - torch.sum(
        (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0).unsqueeze(-1)), 1)
    return loss


# Mutual Information for Dirichlet output
def compute_mutual_information(log_alphas):
    alphas = torch.exp(log_alphas)
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    alpha0 = torch.exp(log_alpha0)
    log_probs = log_alphas - log_alpha0.unsqueeze(-1)
    loss = -torch.sum(torch.exp(log_probs) * (log_probs -
                                              torch.digamma(alphas + 1) +
                                              torch.digamma(alpha0 + 1).unsqueeze(-1)),
                      1)
    return loss


# Precision for Dirichlet output
def compute_precision(log_alphas):
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    return torch.exp(log_alpha0)


# Data Uncertainty for Dirichlet output
def compute_data_uncertainty(log_alphas):
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    log_probs = log_alphas - log_alpha0.unsqueeze(-1)
    alphas = torch.exp(log_alphas)
    alpha0 = torch.exp(log_alpha0)
    loss = - torch.sum(
        log_probs * (torch.digamma(alphas + 1) -
                     torch.digamma(alpha0 + 1).unsqueeze(-1)),
        1)
    return loss

def threshold(preds, tau):
    if isinstance(preds, np.ndarray):
        return np.where(preds>tau, 1.0, 0.0)

    elif torch.is_tensor(preds):
        return torch.where(preds>tau, 1.0, 0.0)

    else:
        raise TypeError(f"ERROR: preds is expected to be of type (torch.tensor, numpy.ndarray) but is type {type(preds)}")

# also known as recall or true positive rate (TPR)
def sensitivity(tp, fn):
    return tp / (tp + fn)

# Also known as selectivity, or true negative rate (TNR)
def specificity(tn, fp):
    return tn / (tn + fp)

# beta > 1 gives more weight to specificity, while beta < 1 favors
# sensitivity. For example, beta = 2 makes specificity twice as important as
# sensitivity, while beta = 0.5 does the opposite.
def f_score_sens_spec(sens, spec, beta=1.0):

    # return (1 + beta**2) * ( (precision * recall) / ( (beta**2 * precision) + recall ) )

    return (1 + beta**2) * ( (sens * spec) / ( (beta**2 * sens) + spec ) )
