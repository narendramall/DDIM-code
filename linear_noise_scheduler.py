import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumpod(self.alphas, dim = 0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1. - self.alpha_cum_prod)

    
    def add_noise(self, original, noise, t):
        original_shape = original.shape # B * C * H * W
        batch_size = original_shape[0] # B

        # t -> 1D tensor of size B
        # we need to calculate -> xt = sqrt(cum prod apha_t) * x0 + sqrt(1. - cum prod apha_t) * Normal Dis. Noise
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        # predict the image with the current image xt
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)

        # mean and variance formula
        ''''
        mean = (1. / sqrt(aplha_t)) * (xt - (1. - alpha_t) / sqrt(1. - cum_prod_alpha_t))

        variance = (1. - alpha_t) * (1. - cum_prod_alpha_t-1) / (1. - cum_prod_alpha_t)

        and then finally xt-1 = mean + sqrt(variance) * noise_pred
        '''

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if (t  == 0):
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod[t-1]) / (1 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]

            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
    
        


    





