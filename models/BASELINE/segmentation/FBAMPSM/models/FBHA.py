import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MFDA(nn.Module):
    def __init__(self, in_channels, levels=3, sigma=1.0):
        super(MFDA, self).__init__()

        self.levels = levels
        self.in_channels = in_channels
        self.sigma = sigma

        self.gaussian_kernels = nn.ParameterList([
            nn.Parameter(self._create_gaussian_kernel(sigma * (2 ** i)), requires_grad=False)
            for i in range(levels)
        ])

        self.frequency_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels, 1),
                nn.Sigmoid()
            ) for _ in range(levels + 1)
        ])

        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(in_channels * (levels + 1), in_channels * 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 1)
        )

        self.output_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

    def _create_gaussian_kernel(self, sigma=1.0, kernel_size=5):
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _gaussian_blur(self, x, kernel):
        kernel = kernel.repeat(self.in_channels, 1, 1, 1)
        return F.conv2d(x, kernel, groups=self.in_channels, padding=kernel.size(-1) // 2)

    def _build_gaussian_pyramid(self, x):
        gaussian_pyramid = [x]
        current = x

        for i in range(self.levels):
            blurred = self._gaussian_blur(current, self.gaussian_kernels[i])
            downsampled = F.avg_pool2d(blurred, 2, stride=2)
            gaussian_pyramid.append(downsampled)
            current = downsampled

        return gaussian_pyramid

    def _build_laplacian_pyramid(self, gaussian_pyramid):
        laplacian_pyramid = []

        for i in range(len(gaussian_pyramid) - 1):
            upsampled = F.interpolate(
                gaussian_pyramid[i + 1],
                size=gaussian_pyramid[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            laplacian = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    def _apply_frequency_attention(self, laplacian_pyramid):
        attended_pyramid = []

        for i, freq_component in enumerate(laplacian_pyramid):
            attention_weights = self.frequency_attentions[i](freq_component)
            attended_component = freq_component * attention_weights
            attended_pyramid.append(attended_component)

        return attended_pyramid

    def _reconstruct_from_pyramid(self, attended_pyramid, target_size):
        reconstructed = attended_pyramid[-1]

        for i in range(len(attended_pyramid) - 2, -1, -1):
            reconstructed = F.interpolate(
                reconstructed,
                size=attended_pyramid[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            reconstructed = reconstructed + attended_pyramid[i]
        return F.interpolate(
            reconstructed,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

    def _multi_scale_feature_fusion(self, attended_pyramid, target_size):
        aligned_features = []
        for freq_component in attended_pyramid:
            aligned = F.interpolate(
                freq_component,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            aligned_features.append(aligned)
        concatenated = torch.cat(aligned_features, dim=1)
        fused = self.multi_scale_fusion(concatenated)

        return fused

    def forward(self, x):

        original_size = x.shape[-2:]
        gaussian_pyramid = self._build_gaussian_pyramid(x)
        laplacian_pyramid = self._build_laplacian_pyramid(gaussian_pyramid)
        attended_pyramid = self._apply_frequency_attention(laplacian_pyramid)
        multi_scale_features = self._multi_scale_feature_fusion(attended_pyramid, original_size)

        combined = torch.cat([x, multi_scale_features], dim=1)
        gate = self.output_gate(combined)

        enhanced_features = x + gate * multi_scale_features

        return enhanced_features


class SBAA(nn.Module):

    def __init__(self, in_channels, boundary_channels=1):
        super(SBAA, self).__init__()

        self.boundary_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, boundary_channels, 1),
            nn.Sigmoid()
        )

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        self.boundary_attention = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
    
        gray = torch.mean(x, dim=1, keepdim=True)

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        boundary_pred = self.boundary_detector(x)

        combined = torch.cat([x, grad_x, grad_y], dim=1)
        boundary_attention = self.boundary_attention(combined)

        enhanced = x * (1 + boundary_attention * boundary_pred)

        return enhanced


class FBHA(nn.Module):

    def __init__(self, in_channels, pyramid_levels=3,
                 use_boundary_attention=True, use_frequency_attention=True):
        super(FBHA, self).__init__()

        self.use_boundary_attention = use_boundary_attention
        self.use_frequency_attention = use_frequency_attention

        if use_frequency_attention:
            self.frequency_attention = MFDA(in_channels, pyramid_levels)


        if use_boundary_attention:
            self.boundary_attention = SBAA(in_channels)

        fusion_channels = in_channels
        if use_frequency_attention: fusion_channels += in_channels
        if use_boundary_attention: fusion_channels += in_channels

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, in_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):

        features = [x]

        if self.use_frequency_attention:
            freq_enhanced = self.frequency_attention(x)
            features.append(freq_enhanced)

        if self.use_boundary_attention:
            boundary_enhanced = self.boundary_attention(x)
            features.append(boundary_enhanced)

        concatenated = torch.cat(features, dim=1)
        fusion_weights = self.feature_fusion(concatenated)

        enhanced = x * fusion_weights
        output = self.output_conv(enhanced)

        return output
