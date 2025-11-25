import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generate_noise(t, theta, k_bar, a, r, x, ntype="Gamma"):
    gaussian_noise = torch.randn_like(x)
    if t == -1:
        eps = torch.zeros_like(x)
    elif ntype == "Gamma":
        concentration = torch.ones(x.size()).to(x.device) * k_bar[j]
        rates = torch.ones(x.size()).to(x.device) * theta[j]
        m = torch.distributions.Gamma(concentration, 1 / rates)
        eps = m.sample()
        eps = eps - concentration * rates
        eps = eps / (1.0 - a[j]).sqrt()
    elif ntype == "Gaussian":
        eps = gaussian_noise
    elif r < 0: # Gamma first |r|t steps of gamma (1-|r|)t gaussian
        s = int((-r)*t)
        concentration_start = torch.ones(x.size()).to(x.device) * k_bar[s]
        rates_start = torch.ones(x.size()).to(x.device) * theta[s]
        m_start = torch.distributions.Gamma(concentration_start, 1 / rates_start)
        e_gamma_start = m_start.sample() - concentration_start*rates_start
        e_gamma_start = e_gamma_start / (1 - a[s])**0.5
        e_gamma_start = ((1 - a[s])**0.5*e_gamma_start*(a[t]/a[s])**0.5 + (1 - a[t]/a[s])**0.5*gaussian_noise) / (1 - a[t])**0.5
        eps = e_gamma_start
    else: #Gaussian
        s = int(r*t)
        concentration_end = torch.ones(x.size()).to(x.device) * (k_bar[t] -k_bar[s])
        rates_end = torch.ones(x.size()).to(x.device) * theta[t]
        m_end = torch.distributions.Gamma(concentration_end, 1 / rates_end)
        e_gamma_end = m_end.sample() - concentration_end*rates_end
        e_gamma_end = e_gamma_end / (1 - a[t]/a[s])**0.5
        e_gamma_end[e_gamma_end==float('inf')]=0
        e_gamma_end = ((1 - a[s])**0.5 * gaussian_noise*(a[t]/a[s])**0.5 + (1 - a[t]/a[s])**0.5*e_gamma_end)/(1 - a[t])**0.5
        eps = e_gamma_end
    return eps


def generalized_concat(x, seq, model, b, r, ntype="Gamma", **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        theta_0 = 0.01
        a = (1 - b).cumprod(dim=0)
        k = (b / a)/theta_0**2
        theta = (a.sqrt()*theta_0)
        k_bar = k.cumsum(dim=0)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            eps = generate_noise(j, theta, k_bar, a, r, x, ntype=ntype).to(x.device)
            xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def generalized_gamma(x, seq, model, b, theta_0=1, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        theta_0 = theta_0
        a = (1 - b).cumprod(dim=0)
        k = (b / a)/theta_0**2
        theta = (a.sqrt()*theta_0)
        k_bar = k.cumsum(dim=0)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t).to(x.device)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            concentration = torch.ones(x.size()).to(x.device) * k_bar[j]
            rates = torch.ones(x.size()).to(x.device) * theta[j]
            m = torch.distributions.Gamma(concentration, 1 / rates)
            eps = m.sample()
            eps = eps - concentration * rates
            eps = eps / (1.0 - a[j]).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
