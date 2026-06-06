import sys
# sys.path.append('/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes')
import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from .base import Base, init_layer, init_bn, act
from .FaSNet import DPRNN
from .gumbel_sigmoid import gumbel_sigmoid
import matplotlib.pyplot as plt


# class FiLM(nn.Module):
#     def __init__(self, film_meta, condition_size):
#         super(FiLM, self).__init__()

#         self.condition_size = condition_size

#         self.modules, _ = self.create_film_modules(
#             film_meta=film_meta, 
#             ancestor_names=[],
#         )
        
#     def create_film_modules(self, film_meta, ancestor_names):

#         modules = {}
       
#         # Pre-order traversal of modules
#         for module_name, value in film_meta.items():

#             if isinstance(value, int):

#                 ancestor_names.append(module_name)
#                 unique_module_name = '->'.join(ancestor_names)

#                 modules[module_name] = self.add_film_layer_to_module(
#                     num_features=value, 
#                     unique_module_name=unique_module_name,
#                 )

#             elif isinstance(value, dict):

#                 ancestor_names.append(module_name)
                
#                 modules[module_name], _ = self.create_film_modules(
#                     film_meta=value, 
#                     ancestor_names=ancestor_names,
#                 )

#             ancestor_names.pop()

#         return modules, ancestor_names

#     def add_film_layer_to_module(self, num_features, unique_module_name):

#         layer = nn.Linear(self.condition_size, num_features)
#         init_layer(layer)
#         self.add_module(name=unique_module_name, module=layer)

#         return layer

#     def forward(self, conditions):
        
#         film_dict = self.calculate_film_data(
#             conditions=conditions, 
#             modules=self.modules,
#         )

#         return film_dict

#     def calculate_film_data(self, conditions, modules):

#         film_data = {}

#         # Pre-order traversal of modules
#         for module_name, module in modules.items():

#             if isinstance(module, nn.Module):
#                 film_data[module_name] = module(conditions)[:, :, None, None]

#             elif isinstance(module, dict):
#                 film_data[module_name] = self.calculate_film_data(conditions, module)

#         return film_data


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
        # has_film,
    ):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        # self.has_film = has_film

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    # def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x


class EncoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        momentum: float,
        # has_film,
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum
            # in_channels, out_channels, kernel_size, momentum, has_film,
        )
        self.downsample = downsample

    # def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        # encoder = self.conv_block1(input_tensor, film_dict['conv_block1'])
        encoder = self.conv_block1(input_tensor)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        momentum: float,
        # has_film,
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, kernel_size, momentum, # has_film,
        )
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        # self.has_film = has_film

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor,  
        # self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor, film_dict: Dict,
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        # b1 = film_dict['beta1']

        # b1 = film_dict['beta1']
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor)))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        # x = self.conv_block2(x, film_dict['conv_block2'])
        x = self.conv_block2(x)
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x

class SafeBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, clip_std=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.clip_std = clip_std  # 可选: 限制标准差上限
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        # x: [B, C, H, W]
        if self.training:
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)

            # 避免 var 为 0
            var = torch.clamp(var, min=self.eps)

            # 可选：限制 std 上限
            if self.clip_std is not None:
                std = torch.sqrt(var)
                std = torch.clamp(std, max=self.clip_std)
                var = std ** 2

            x = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                x = x * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)
        else:
            # eval 模式直接用 BatchNorm 自带的 running_mean/var
            x = self.bn(x)
        return x

class SealGPT_Sep_Base(nn.Module, Base):
    def __init__(self, input_channels=1, output_channels=2, dprnn=True, dprnn_layers=2, dprnn_hidden=128):
        super(SealGPT_Sep_Base, self).__init__()

        self.dprnn = dprnn
        self.dprnn_layers = dprnn_layers
        self.dprnn_hidden = dprnn_hidden

        window_size = 1024 # 2048
        hop_size = 160 # 10 ms for 16000hz audio # 320
        center = True 
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum, eps=1e-5)
        # self.norm = nn.InstanceNorm2d(num_features=window_size // 2 + 1, affine=True, eps=1e-5)
        # self.bn0 = SafeBatchNorm2d(num_features=window_size // 2 + 1, eps=1e-5, clip_std=10.0)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        # # ====== decoder for sound ======
        self.decoder_block1_1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
        )
        self.decoder_block2_1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block3_1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block4_1 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block5_1 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block6_1 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )

        self.after_conv_1 = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        # ===================

        self.init_weights()
        # --- hanyin ---
        if self.dprnn:
            self.DPRNN = nn.Sequential(DPRNN('transformer_encoder', 384, self.dprnn_hidden, 384, dropout = 0.1,
                                             num_layers=self.dprnn_layers))
        # --- han yin ---
        
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

    # def forward(self, mixtures, film_dict):
    def forward(self, mixtures):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag
        # x = torch.log1p(mag)
        # x = 20 * torch.log10(mag + 1e-6)
        # print(mixtures.shape) # torch.Size([6, 1, 160000])
        # print("mixtures: ", mixtures)
        # print("mag: ", mag)
        # print("mag: ", mag.max(), mag.min(), mag.mean(), mag.std())
        # print("mag: ", torch.isnan(mag).any(), torch.isinf(mag).any())

        # Batch normalization
        x = x.transpose(1, 3)
        # print("x1:", x)
        # print("x1: max:", x.max(), "min:", x.min(), "mean:", x.mean(), "std:", x.std(), "shape: ", x.shape)
        # print("x1:", torch.isnan(x).any(), torch.isinf(x).any())
        # fig = x[0, :, :, 0].detach().cpu().numpy()
        # x_before_bn = x
        # print("running_mean:", self.bn0.running_mean.mean())
        # print("running_var:", self.bn0.running_var.mean())
        x = self.bn0(x)
        # print("x2:", x)
        if not torch.isnan(x).any():
            pass
            # plt.figure(figsize=(12, 6))
            # plt.imshow(fig, aspect='auto', origin='lower', cmap='magma')  # 'magma' 好看一点
            # plt.colorbar(format="%+2.0f")  # 如果你已经做过 log，可以改掉 dB
            # plt.xlabel("Time Frames")
            # plt.ylabel("Frequency Bins")
            # plt.title("Spectrogram")
            # plt.savefig("/home/han/projects/seal_gpt/seal_gpt_separator/pics/debug_normal.png", dpi=600)
        else:
            print("after bn there are nan values ...")
            # print("bn0 input:", x_before_bn.mean(), x_before_bn.std(), torch.isnan(x_before_bn).any())
            print("bn0.running_mean:", self.bn0.running_mean.mean())
            print("bn0.running_var:", self.bn0.running_var.mean())
            print("bn0.weight (gamma):", self.bn0.weight.abs().max())

            # plt.figure(figsize=(12, 6))
            # plt.imshow(fig, aspect='auto', origin='lower', cmap='magma')  # 'magma' 好看一点
            # plt.colorbar(format="%+2.0fdB")  # 如果你已经做过 log，可以改掉 dB
            # plt.xlabel("Time Frames")
            # plt.ylabel("Frequency Bins")
            # plt.title("Spectrogram")
            # plt.savefig("/home/han/projects/seal_gpt/seal_gpt_separator/pics/debug_abnormal.png", dpi=600)

        x = x.transpose(1, 3)
        # print("x3:", x)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # print("x5:", x) # nan
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)
        # print("x6:", x) # nan
        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool)  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool)  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool)  # (bs, 384, T / 32, F / 64)
        
        # DPRNN Block
        if self.dprnn:
            x_center = self.DPRNN(x_center)
        # # # 

        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_speech = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        # p = self.class_cnn(x_center) # [batch, 1]
        # p = p.view(p.shape[0], -1)
        # # print("x_center: ", x_center) # nan
        # # print("p: ", p) # nan
        # # print("self.class_fc(p): ", self.class_fc(p))
        # p_soft = self.sigmoid(self.class_fc(p))

        # print(p_speech.shape, p_sound.shape)

        # ===================== sound decoder branch =================
        # # decoding sound
        x7_1 = self.decoder_block1_1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8_1 = self.decoder_block2_1(x7_1, x5)  # (bs, 384, T / 16, F / 16)
        x9_1 = self.decoder_block3_1(x8_1, x4)  # (bs, 256, T / 8, F / 8)
        x10_1 = self.decoder_block4_1(x9_1, x3)  # (bs, 128, T / 4, F / 4)
        x11_1 = self.decoder_block5_1(x10_1, x2)  # (bs, 64, T / 2, F / 2)
        x12_1 = self.decoder_block6_1(x11_1, x1)  # (bs, 32, T, F)

        x_1 = self.after_conv_1(x12_1)
        x_1 = x_1[:, :, :origin_len, :]

        # Recover shape
        x_1 = F.pad(x_1, pad=(0, 1))
        x_1 = x_1[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        # print(x.shape, x_1.shape)
        separated_sound = self.feature_maps_to_wav(
            input_tensor=x_1,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )

        output_dict = {
            'pred_speech': separated_speech,
            'pred_sound': separated_sound,
        }

        return output_dict


# def get_film_meta(module):

#     film_meta = {}

#     if hasattr(module, 'has_film'):\

#         if module.has_film:
#             film_meta['beta1'] = module.bn1.num_features
#             film_meta['beta2'] = module.bn2.num_features
#         else:
#             film_meta['beta1'] = 0
#             film_meta['beta2'] = 0

#     for child_name, child_module in module.named_children():

#         child_meta = get_film_meta(child_module)

#         if len(child_meta) > 0:
#             film_meta[child_name] = child_meta
    
#     return film_meta


class SealGPT_Sep(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, # condition_size, 
                 dprnn=True, dprnn_layers=2, dprnn_hidden=128):
        super(SealGPT_Sep, self).__init__()

        self.base = SealGPT_Sep_Base(
            input_channels=input_channels, 
            output_channels=output_channels,
            dprnn=dprnn, 
            dprnn_layers=dprnn_layers, 
            dprnn_hidden=dprnn_hidden
        )
        
        # self.film_meta = get_film_meta(
        #     module=self.base,
        # )
        
        # self.film = FiLM(
        #     film_meta=self.film_meta, 
        #     condition_size=condition_size
        # )


    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        # conditions = input_dict['condition']

        # film_dict = self.film(
        #     conditions=conditions,
        # )

        output_dict = self.base(
            mixtures=mixtures, 
            # film_dict=film_dict,
        )

        return output_dict

    @torch.no_grad()
    def chunk_inference(self, input_dict, NL=1.0, NC=3.0, NR=1.0, sr=16000):
        # input must be single sample, not for batch inference
        chunk_config = {
                    'NL': NL,
                    'NC': NC,
                    'NR': NR,
                    'RATE': sr # 32000
                }

        mixtures = input_dict['mixture']
        # conditions = input_dict['condition']

        # film_dict = self.film(
        #     conditions=conditions,
        # )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([2, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

        
            chunk_out_data = self.base(
                mixtures=chunk_in, 
                # film_dict=film_dict,
            )
            chunk_out_speech = chunk_out_data['pred_speech']
            chunk_out_sound = chunk_out_data['pred_sound']
            chunk_out = torch.cat([chunk_out_speech, chunk_out_sound], dim=1)
            # print(chunk_out.shape) # torch.Size([1, 2, 80000])
            

            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            # print(chunk_out_np.shape) # (1, 2, 80000)

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out_data = self.base(
                mixtures=chunk_in, 
                # film_dict=film_dict,
            )
                chunk_out_speech = chunk_out_data['pred_speech']
                chunk_out_sound = chunk_out_data['pred_sound']
                chunk_out = torch.cat([chunk_out_speech, chunk_out_sound], dim=1)

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np


# # single single_decoder
class SealGPT_Sep_Base_single_decoder(nn.Module, Base):
    def __init__(self, input_channels=1, output_channels=2, dprnn=True, dprnn_layers=2, dprnn_hidden=128):
        super(SealGPT_Sep_Base_single_decoder, self).__init__()

        self.dprnn = dprnn
        self.dprnn_layers = dprnn_layers
        self.dprnn_hidden = dprnn_hidden

        window_size = 1024 # 2048
        hop_size = 160 # 10 ms for 16000hz audio # 320
        center = True 
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum, eps=1e-5)
        # self.norm = nn.InstanceNorm2d(num_features=window_size // 2 + 1, affine=True, eps=1e-5)
        # self.bn0 = SafeBatchNorm2d(num_features=window_size // 2 + 1, eps=1e-5, clip_std=10.0)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            # has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()
        # --- hanyin ---
        if self.dprnn:
            self.DPRNN = nn.Sequential(DPRNN('transformer_encoder', 384, self.dprnn_hidden, 384, dropout = 0.1,
                                             num_layers=self.dprnn_layers))
        # --- han yin ---

        
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

    def normalize_waveform(self, waveforms):
        """
        waveforms: torch.Tensor of shape [batch, 1, samples]
        return: normalized_waveforms, max_abs
        """
        # 计算每条 waveform 的最大绝对值
        max_abs = waveforms.abs().amax(dim=2, keepdim=True)  # [batch, 1, 1]
        max_abs[max_abs == 0] = 1.0  # 防止除以0
        normalized = waveforms / max_abs
        return normalized, max_abs
    
    def denormalize_waveform(self, normalized_waveforms, max_abs):
        """
        normalized_waveforms: torch.Tensor of shape [batch, 1, samples], in [-1, 1]
        max_abs: torch.Tensor of shape [batch, 1, 1]
        return: original scale waveform
        """
        return normalized_waveforms * max_abs

    # def forward(self, mixtures, film_dict):
    def forward(self, mixtures):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        # x = mag
        # x = torch.log1p(mag)
        x = 20 * torch.log10(mag + 1e-6)
        # print(mixtures.shape) # torch.Size([6, 1, 160000])
        # print("mixtures: ", mixtures)
        # print("mag: ", mag)
        # print("mag: ", mag.max(), mag.min(), mag.mean(), mag.std())
        # print("mag: ", torch.isnan(mag).any(), torch.isinf(mag).any())

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # print("x5:", x) # nan
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)
        # print("x6:", x) # nan
        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool)  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool)  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool)  # (bs, 384, T / 32, F / 64)
        
        # DPRNN Block
        if self.dprnn:
            x_center = self.DPRNN(x_center)
        # # # 

        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)
        return separated_audio






if __name__ == '__main__':
    from utils import load_ss_model
    device = 'cuda:0'
    resnet = ResUNet30(input_channels=1, output_channels=1, condition_size=512).to(device)
    resnet.eval()
    
    audio = torch.rand((1, 1, 32000 * 5)).to(device)
    caption = torch.ones((1, 512)).to(device)
    input_dict = {'mixture': audio,
                  'condition': caption}
    with torch.no_grad():
        output = resnet(input_dict)
        sep_audio = output['waveform']
        print(sep_audio.shape)