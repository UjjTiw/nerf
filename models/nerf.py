import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, in_channels: int, N_freqs: int, logscale: bool = True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

        self.freq_bands = nn.Parameter(self.freq_bands, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self,
                 D: int = 8,
                 W: int = 256,
                 in_channels_xyz: int = 63,
                 in_channels_dir: int = 27,
                 skips: list = [4]):
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        self.xyz_encodings = nn.ModuleList()
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            self.xyz_encodings.append(nn.Sequential(layer, nn.ReLU(True)))
        
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2),
            nn.ReLU(True)
        )

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
            nn.Linear(W // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, sigma_only: bool = False) -> torch.Tensor:
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i, layer in enumerate(self.xyz_encodings):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], dim=-1)
            xyz_ = layer(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], dim=-1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        return torch.cat([rgb, sigma], dim=-1)

# Example usage
if __name__ == "__main__":
    embedding = Embedding(3, 10)
    nerf = NeRF()
    
    # Test Embedding
    x = torch.randn(100, 3)
    embedded = embedding(x)
    print(f"Embedding output shape: {embedded.shape}")
    
    # Test NeRF
    x = torch.randn(100, 90)  # 63 + 27 = 90
    output = nerf(x)
    print(f"NeRF output shape: {output.shape}")
    
    # Test NeRF with sigma_only
    x = torch.randn(100, 63)
    sigma = nerf(x, sigma_only=True)
    print(f"NeRF sigma_only output shape: {sigma.shape}")
