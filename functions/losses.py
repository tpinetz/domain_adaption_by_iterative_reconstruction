import torch
from torch.profiler import profile, ProfilerActivity


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def poisson_estimation_loss(model,
                            x0: torch.Tensor,
                            t: torch.LongTensor,
                            b: torch.Tensor,
                            factor=1,
                            keepdim=False):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    rates = torch.ones(x0.size()).to(x0.device)* factor * (1 - a)
    e = (torch.poisson(rates) - rates) / ((1.0 - a).sqrt() * factor ** 0.5)
    x = a.sqrt()*x0 + e*(1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def laplace_estimation_loss(model,
                            x0: torch.Tensor,
                            t: torch.LongTensor,
                            b: torch.Tensor,
                            factor=1,
                            keepdim=False):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    scale = (torch.ones(x0.size()) * (1 - a).sqrt()) / (2) ** 0.5
    mean = torch.zeros(x0.size()).to(x0.device)
    m = torch.distributions.Laplace(mean, scale)
    e = m.sample() / (1.0 - a).sqrt()
    x = a.sqrt()*x0 + e*(1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def gamma_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          b: torch.Tensor,
                          theta_0=1,
                          keepdim=False):
    a = (1 - b).cumprod(dim=0)
    k = (b / a)/theta_0**2
    theta = (a.sqrt()*theta_0).index_select(0, t).view(-1, 1, 1, 1)
    k_bar = k.cumsum(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    a = a.index_select(0, t).view(-1, 1, 1, 1)
    concentration = torch.ones(x0.size()).to(x0.device) * k_bar
    rates = torch.ones(x0.size()).to(x0.device) * theta
    m = torch.distributions.Gamma(concentration, 1 / rates)
    e = m.sample()
    e = e - concentration * rates
    e = e / (1.0 - a).sqrt()
    x = a.sqrt()*x0 + e*(1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def concat_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          b: torch.Tensor,
                          theta_0=1,
                          keepdim=False):
    a = (1 - b).cumprod(dim=0)
    k = (b / a) / theta_0 ** 2

    s = (torch.rand(t.size())*t).long()  # intermediate index sampling in [0, t-1]

    theta = (a.sqrt() * theta_0)
    theta_t = theta.index_select(0, t).view(-1, 1, 1, 1)
    theta_s = theta.index_select(0, s).view(-1, 1, 1, 1)
    k_bar = k.cumsum(dim=0)
    k_bar_s = k_bar.index_select(0, s).view(-1, 1, 1, 1)
    k_bar_t = k_bar.index_select(0, t).view(-1, 1, 1, 1)
    k_bar_s_t = k_bar_t - k_bar_s
    a_s = a.index_select(0, s).view(-1, 1, 1, 1)
    a_t = a.index_select(0, t).view(-1, 1, 1, 1)
    a_s_t = a_t / a_s
    concentration_start = torch.ones(x0.size()).to(x0.device) * k_bar_s
    rates_start = torch.ones(x0.size()).to(x0.device) * theta_s
    m_start = torch.distributions.Gamma(concentration_start, 1 / rates_start)
    e_gamma_start = m_start.sample() - concentration_start*rates_start
    e_gamma_start = e_gamma_start / (1 - a_s).sqrt()
    concentration_end = torch.ones(x0.size()).to(x0.device) * k_bar_s_t
    rates_end = torch.ones(x0.size()).to(x0.device) * theta_t
    m_end = torch.distributions.Gamma(concentration_end, 1 / rates_end)
    e_gamma_end = m_end.sample() - concentration_end*rates_end
    e_gamma_end = e_gamma_end / (1 - a_s_t).sqrt()
    e_gamma_end[e_gamma_end == float('inf')] = 0
    gaussian_noise = torch.randn_like(x0)
    e_gamma_end = ((1 - a_s).sqrt()*gaussian_noise*a_s_t.sqrt() + (1-a_s_t).sqrt()*e_gamma_end)/(1 - a_t).sqrt()
    e_gamma_start = ((1 - a_s).sqrt()*e_gamma_start*a_s_t.sqrt() + (1-a_s_t).sqrt()*gaussian_noise)/(1 - a_t).sqrt()

    order = torch.randint(2, size=s.size()).view(-1, 1, 1, 1) # if 1 gaussian first else gamma first
    e = e_gamma_end*order + e_gamma_start*(1-order)
    x = a_t.sqrt() * x0 + e * (1.0 - a_t).sqrt()
    output = model(x, t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def gauss2_estimation_loss(model,
                           x0: torch.Tensor,
                           t: torch.LongTensor,
                           b: torch.Tensor,
                           sigma: torch.Tensor,
                           p=0.5,
                           keepdim=False):
    sigma_t = sigma.index_select(0, t).view(-1, 1, 1, 1)
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    m1_t = ((1 - sigma_t ** 2) / (p * (1 - p) + p ** 3 / (1 - p) + 2 * p ** 2)) ** 0.5
    m2_t = - (p / (1 - p)) * m1_t
    y_1 = torch.randn_like(x0) * sigma_t + m1_t
    y_2 = torch.randn_like(x0) * sigma_t + m2_t
    b = torch.randint(2, size=x0.size()).to(x0.device)
    e = b*y_1 + (1 - b)*y_2
    x = a.sqrt() * x0 + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
    'poisson': poisson_estimation_loss,
    'laplace': laplace_estimation_loss,
    'gamma': gamma_estimation_loss,
    'concat': concat_estimation_loss,
    '2gauss': gauss2_estimation_loss,
}
