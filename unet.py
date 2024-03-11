import torch
import torch.nn as nn

def get_time_embedding(time_steps, t_emb_dim):
    # time_steps -> B sized 1 d tensor timestep
    # t_emb_dim = scalar value, embedding dims

    # return 
    # B * t_emb_dim tensor

    # formula
    '''
    sin(pos / (10000 ** (2 * i / d_model)))
    cos(pos / (10000 ** (2 * i / d_model)))
    '''

    factor = 10000 ** ((torch.arange(
        start = 0, end = t_emb_dim//2, device = time_steps.device
    ), (t_emb_dim // 2)))

    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)

    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample # whether or not downsample -> a boolean
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2D(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )

        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )

        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        )

        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first = True)

        self.residual_input_conv = nn.Conv2D(in_channels, out_channels, kernel_size = 1)
        # for downsampling one can use pooling layer as well
        self.down_sample_conv = nn.Conv2D(out_channels, 
                                          out_channels, 
                                          kernel_size = 4, 
                                          stride = 2, 
                                          padding=1
                                          ) if self.down_sample else nn.Identity()
        
        def forward(self, x, t_emb):
            
