from .FaSNet import DPRNN
from .base import Base, init_layer, init_bn, act
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, NoReturn, Tuple
import numpy as np
from .gumbel_sigmoid import gumbel_sigmoid

class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
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

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

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
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
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
            out_channels * 2, out_channels, kernel_size, momentum
        )
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        
        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor)))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        x = self.conv_block2(x)
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x

class LearnableShiftedSigmoid(nn.Module):
    def __init__(self, init_alpha=1.0): 
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        
    def forward(self, x, scale = True):
        if scale:
            k = 1.0 + F.softplus(self.alpha) 
        else: # normal sigmoid
            k = 1.0
        return k * torch.sigmoid(x)

class SealGPT_Sep(nn.Module, Base):
    def __init__(self, input_channels=1, output_channels=1, n_mel=128, dprnn=True, dprnn_layers=2, dprnn_hidden=128, momentum=0.01):
        super(SealGPT_Sep, self).__init__()

        self.dprnn = dprnn
        self.dprnn_layers = dprnn_layers
        self.dprnn_hidden = dprnn_hidden

        self.n_mel = n_mel

        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.bn0 = nn.BatchNorm2d(n_mel, momentum=momentum)
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
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
        )

        # # ====== decoder for speech ======
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels,
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
            out_channels=output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        # ======

        self.init_weights()
        
        if self.dprnn:
            self.DPRNN = nn.Sequential(DPRNN('transformer_encoder', 384, self.dprnn_hidden, 384, dropout = 0.1,
                                             num_layers=self.dprnn_layers))

        self.lr_sigmoid = LearnableShiftedSigmoid()
        self.lr_sigmoid_1 = LearnableShiftedSigmoid()

        # classifier
        self.class_cnn_speech = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(3,1), padding='same'), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.class_fc_speech = nn.Linear(64, 1)
        
        self.class_cnn_sound = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(3,1), padding='same'), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
            )
        self.class_fc_sound = nn.Linear(64, 1)
        

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def forward(self, mixture):
        """
        Args:
          input: (batch_size, channel, time_steps, freq_bins)

        Outputs:
          speech_mel: (batch_size, channel, time_steps, freq_bins)
          sound_mel: (batch_size, channel, time_steps, freq_bins)
        """

        # Batch normalization
        x = mixture.transpose(1, 3)
        # print("x1: max:", x.max(), "min:", x.min(), "mean:", x.mean(), "std:", x.std(), "shape: ", x.shape)
        # print("x1:", torch.isnan(x).any(), torch.isinf(x).any())
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""
        # print(f"x: {x.shape}") # x: torch.Size([4, 1, 3000, 128])

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""
        # print(f"padded x: {x.shape}") padded x: torch.Size([4, 1, 3008, 128])

        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool)  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool)  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool)  # (bs, 384, T / 32, F / 64)
        # print(f"x_center: {x_center.shape}") x_center: torch.Size([4, 384, 94, 2])
        
        # DPRNN Block
        if self.dprnn:
            x_center = self.DPRNN(x_center)
        # # # 
        # print(f"x_center after dprnn: {x_center.shape}") # x_center after dprnn: torch.Size([4, 384, 94, 2])

        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)

        x = self.after_conv(x12)
        # print(f"x after decoder: {x.shape}") x after decoder: torch.Size([4, 2, 3008, 128])
        # Recover shape
        x = x[:, :, :origin_len, :]

        # print(f"x recovered: {x.shape}") x recovered: torch.Size([4, 2, 3000, 128])
        speech_mask = self.lr_sigmoid(x, scale = True)

        # # decoding sound
        x7_1 = self.decoder_block1_1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8_1 = self.decoder_block2_1(x7_1, x5)  # (bs, 384, T / 16, F / 16)
        x9_1 = self.decoder_block3_1(x8_1, x4)  # (bs, 256, T / 8, F / 8)
        x10_1 = self.decoder_block4_1(x9_1, x3)  # (bs, 128, T / 4, F / 4)
        x11_1 = self.decoder_block5_1(x10_1, x2)  # (bs, 64, T / 2, F / 2)
        x12_1 = self.decoder_block6_1(x11_1, x1)  # (bs, 32, T, F)

        x_1 = self.after_conv_1(x12_1)
        x_1 = x_1[:, :, :origin_len, :]
        sound_mask = self.lr_sigmoid_1(x_1, scale = True)

        # # ======
        # speech_mask, sound_mask = mask[:, 0, :, :].unsqueeze(1), mask[:, 1, :, :].unsqueeze(1)
        # print(speech_mask.shape, sound_mask.shape) # torch.Size([4, 1, 3000, 128]) torch.Size([4, 1, 3000, 128])
        
        # classcify
        p_speech = self.class_cnn_speech(x_center) # [batch, 1]
        # print(p_speech.shape)
        p_speech = p_speech.view(p_speech.shape[0], -1)
        # print(p_speech.shape)
        p_speech = self.class_fc_speech(p_speech)
        # print(p_speech.shape)

        p_sound = self.class_cnn_sound(x_center) # [batch, 1]
        p_sound = p_sound.view(p_sound.shape[0], -1)
        p_sound = self.class_fc_sound(p_sound)

        # print(p_speech.shape, p_sound.shape)
        p_speech_sig_soft = gumbel_sigmoid(p_speech, hard=False)
        # p_speech_sig_soft, p_speech_sig_hard = gumbel_sigmoid(p_speech, hard=True)
        # print(p_speech_sig_soft, p_speech_sig_hard)

        p_sound_sig_soft = gumbel_sigmoid(p_sound, hard=False)
        # p_sound_sig_soft, p_sound_sig_hard = gumbel_sigmoid(p_sound, hard=True)
        # print(p_sound_sig_soft, p_sound_sig_hard)
        

        # speech_mel = (mixture * speech_mask) * p_speech_sig_hard[:, :, None, None]
        # sound_mel = (mixture * sound_mask) * p_sound_sig_hard[:, :, None, None]
        speech_mel = (mixture * speech_mask)
        sound_mel = (mixture * sound_mask)

        return speech_mel, sound_mel, p_speech_sig_soft, p_sound_sig_soft