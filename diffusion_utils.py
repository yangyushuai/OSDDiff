import torch
from torch.nn import functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 8
all_time_steps = 1000  # 扩散1000步
epochs = 200
ensembles_num = 10

betas = torch.linspace(1e-4, 0.02, all_time_steps)
alphas = 1. - betas
sqrt_alphas_recip = torch.sqrt(1. / alphas)
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_prev = torch.cat([torch.tensor([1.]), alphas_bar[:-1]], dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)  # 后验方差
sqrt_posterior_variance = torch.sqrt(posterior_variance)


def get_index_element(arr, t):
    bs = t.shape[0]
    out = arr[t]
    return out.reshape(bs, 1, 1, 1).to(device)


def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alphas_bar_t = get_index_element(sqrt_alphas_bar, t)
    sqrt_one_minus_alphas_bar_t = get_index_element(sqrt_one_minus_alphas_bar, t)
    x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise
    return x_t, noise


def compute_losses(denoise_model, x, label, t):
    x_t, noise = q_sample(label, t)
    x_in = torch.concat([x_t, x], dim=1)
    pred_noise = denoise_model(x_in, t)
    mse_loss = F.mse_loss(noise, pred_noise)

    return mse_loss


def p_sample(denoise_model, img, x, t, index):
    sqrt_alphas_recip_t = get_index_element(sqrt_alphas_recip, t)
    betas_t = get_index_element(betas, t)
    sqrt_one_minus_alphas_bar_t = get_index_element(sqrt_one_minus_alphas_bar, t)
    img_in = torch.concat([img, x], dim=1)
    mean_t = sqrt_alphas_recip_t * (img - betas_t * denoise_model(img_in, t) / sqrt_one_minus_alphas_bar_t)
    if index == 0:
        return mean_t
    else:
        posterior_variance_t = get_index_element(sqrt_posterior_variance, t)
        noise = torch.randn_like(img)
        return mean_t + posterior_variance_t * noise
