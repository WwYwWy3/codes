import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import copy
from mamba_ssm import Mamba

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        self.mamba_forw = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)
        self.mamba_backw = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        
        self.mamba_spiral = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.mamba_zigzag = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.mamba_block = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
        self.mamba_spiral.out_proj = nn.Identity()
        self.mamba_zigzag.out_proj = nn.Identity()
        self.mamba_block.out_proj = nn.Identity()
        
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5)
        
        
    def generate_spiral_indices(self, height, width):
        indices = []
        visited = torch.zeros(height, width, dtype=torch.bool)
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction_idx = 0
        
        row, col = 0, 0
        
        for _ in range(height * width):
            indices.append(row * width + col)
            visited[row, col] = True
            
            dr, dc = directions[direction_idx]
            next_row, next_col = row + dr, col + dc
            
            if (next_row < 0 or next_row >= height or 
                next_col < 0 or next_col >= width or 
                visited[next_row, next_col]):
                direction_idx = (direction_idx + 1) % 4
                dr, dc = directions[direction_idx]
                next_row, next_col = row + dr, col + dc
            
            row, col = next_row, next_col
        
        return torch.tensor(indices, dtype=torch.long)
    
    def generate_zigzag_indices(self, height, width):
        indices = []
        
        for diagonal in range(height + width - 1):
            diagonal_indices = []
            
            if diagonal < width:
                start_row, start_col = 0, diagonal
            else:
                start_row, start_col = diagonal - width + 1, width - 1
            
            row, col = start_row, start_col
            while row < height and col >= 0:
                diagonal_indices.append(row * width + col)
                row += 1
                col -= 1
            
            if diagonal % 2 == 1:
                diagonal_indices.reverse()
            
            indices.extend(diagonal_indices)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def generate_block_indices(self, height, width, block_size=2):
        indices = []
        
        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size
        
        for block_row in range(blocks_h):
            for block_col in range(blocks_w):
                start_h = block_row * block_size
                end_h = min(start_h + block_size, height)
                start_w = block_col * block_size
                end_w = min(start_w + block_size, width)
                
                for h in range(start_h, end_h):
                    for w in range(start_w, end_w):
                        indices.append(h * width + w)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def apply_scanning_pattern(self, x_flat, indices, device):
        NB, n_tokens, C = x_flat.shape
        indices = indices.to(device)
        
        indices = indices[indices < n_tokens]
        
        if len(indices) < n_tokens:
            remaining = torch.tensor([i for i in range(n_tokens) if i not in indices], 
                                   dtype=torch.long, device=device)
            indices = torch.cat([indices, remaining])
        
        x_reordered = x_flat[:, indices, :]
        return x_reordered, indices
    
    def spiral_scan(self, x_norm, img_dims, device):
        if len(img_dims) == 2: 
            height, width = img_dims
            spiral_indices = self.generate_spiral_indices(height, width)
            x_spiral, indices = self.apply_scanning_pattern(x_norm, spiral_indices, device)
            y_spiral = self.mamba_spiral(x_spiral)
            
            inverse_indices = torch.argsort(indices).to(device)
            y_spiral_restored = y_spiral[:, inverse_indices, :]
            return y_spiral_restored
        else:
            return self.mamba_spiral(x_norm)
    
    def zigzag_scan(self, x_norm, img_dims, device):

        if len(img_dims) == 2:
            height, width = img_dims
            zigzag_indices = self.generate_zigzag_indices(height, width)
            x_zigzag, indices = self.apply_scanning_pattern(x_norm, zigzag_indices, device)
            y_zigzag = self.mamba_zigzag(x_zigzag)
            
            inverse_indices = torch.argsort(indices).to(device)
            y_zigzag_restored = y_zigzag[:, inverse_indices, :]
            return y_zigzag_restored
        else:
            return self.mamba_zigzag(x_norm)
    
    def block_scan(self, x_norm, img_dims, device):

        if len(img_dims) == 2:
            height, width = img_dims

            block_size = 2 if min(height, width) >= 4 else 1
            block_indices = self.generate_block_indices(height, width, block_size)
            x_block, indices = self.apply_scanning_pattern(x_norm, block_indices, device)
            y_block = self.mamba_block(x_block)
            
            inverse_indices = torch.argsort(indices).to(device)
            y_block_restored = y_block[:, inverse_indices, :]
            return y_block_restored
        else:
            return self.mamba_block(x_norm)
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        B, C = x.shape[:2]
        assert C == self.dim
        img_dims = x.shape[2:]
        device = x.device
        
        x_div = x
        NB = x_div.shape[0]
        n_tokens = x_div.shape[2:].numel()
        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)  # [NB,n_tokens,C]
        x_norm = self.norm(x_flat)
        
        y_norm = torch.flip(x_norm, dims=[1])
        x_mamba_forw = self.mamba_forw(x_norm)
        x_mamba_backw = self.mamba_backw(y_norm)
        x_mamba_backw = torch.flip(x_mamba_backw, dims=[1])
        
        x_mamba_spiral = self.spiral_scan(x_norm, img_dims, device)
        x_mamba_zigzag = self.zigzag_scan(x_norm, img_dims, device)
        x_mamba_block = self.block_scan(x_norm, img_dims, device)
        
        fusion_weights = torch.softmax(self.fusion_weights, dim=0)
        conv111 = nn.Conv1d(5 * img_dims[0] * img_dims[1], img_dims[0] *img_dims[1], 1).cuda()
        
        x_mamba_fused = (fusion_weights[0] * x_mamba_forw + 
                        fusion_weights[1] * x_mamba_backw  +
                        fusion_weights[2] * x_mamba_spiral +
                        fusion_weights[3] * x_mamba_zigzag +
                        fusion_weights[4] * x_mamba_block)
        
        x_out = self.out_proj(x_mamba_fused)
        x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        
        out = x_out + x
        return out


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.B_SS2D = MambaLayer(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.B_SS2D(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class MPSM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = Block(in_channels)

    def forward(self, x):
        x1 = self.block(x)
        return x1