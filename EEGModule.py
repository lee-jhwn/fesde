import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from s4_block.s4_model import S4Model


class block_conv(nn.Module):
    def __init__(self, in_channel: int=512, out_channel: int=512, 
                 kernel_size: int=3, stride: int=1, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, conv: str='conv'):
        '''A single convolution block with normaliation and dropout'''
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dropout = dropout

        if residual is not False:
            raise NotImplementedError

        if conv == 'conv':
            self.conv_layer = nn.Conv1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding)
        else:
            self.conv_layer = nn.ConvTranspose1d(in_channels=self.in_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding,
                                        output_padding=self.output_padding)
        
        self.norm_layer = norm(1, out_channel)

        self.dropout_layer = nn.Dropout1d(self.dropout)

        if act_layer is not None:
            self.act_layer = act_layer()
        else:
            self.act_layer = None

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout_layer(x)
        x = self.norm_layer(x)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x

class block_last_decoder(block_conv):
    def __init__(self, in_channel: int = 512, out_channel: int = 512, kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 1, norm=nn.GroupNorm, dropout: float = 0.3, residual: bool = False, act_layer=nn.GELU, conv: str = 'conv'):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, output_padding, norm, dropout, residual, act_layer, conv)
    
    def forward(self, x):
        x = self.conv_layer(x)
        return x
    
class conv_encoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512, 
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()

        self.block_layers.append(block_conv(input_channels, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))
        
        for i in range(n_layers - 1):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer))

        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x
    
class conv_decoder_tueg(nn.Module):
    def __init__(self, n_layers: int=4, input_channels: int=19, embedding_size: int=512, 
                 kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
                 norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
                 act_layer = nn.GELU, is_last_layer=False):
        super().__init__()

        self.n_layers = n_layers

        self.block_layers = nn.ModuleList()
        
        for i in range(n_layers - 1 * is_last_layer):
            self.block_layers.append(block_conv(embedding_size, embedding_size,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, act_layer, 'deconv'))
        
        if is_last_layer:
            self.block_layers.append(block_last_decoder(embedding_size, input_channels,
                                            kernel_size, stride, padding, output_padding,
                                            norm, dropout, residual, None, 'deconv'))
        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
        return x



class EEGModule(nn.Module):
    def __init__(self, n_layers_cnn: int=6,
                 use_s4=False,
                n_layers_s4: int=8,
                 device: str='cuda', embedding_size: int=512,
                 is_mask: bool=True, in_channels: int=19):
        super().__init__()

        self.n_layers_cnn = n_layers_cnn
        self.use_s4 = use_s4
        self.n_layers_s4 = n_layers_s4
        self.device = device
        self.is_mask = is_mask

        self.conv_encoder = conv_encoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, embedding_size=embedding_size, input_channels=in_channels)
        self.conv_encoder2 = conv_encoder_tueg(1, stride=3, padding=2, kernel_size=4, embedding_size=embedding_size, input_channels=embedding_size)

        self.deconv_encoder = conv_decoder_tueg(n_layers_cnn, stride=1, padding=3, kernel_size=4, output_padding=0, embedding_size=embedding_size, input_channels=in_channels, is_last_layer=True)
        self.deconv_encoder2 = conv_decoder_tueg(1, stride=3, padding=2, kernel_size=4, output_padding=2, embedding_size=embedding_size, input_channels=embedding_size, is_last_layer=False)

        self.s4_model = S4Model(d_input=embedding_size,
        d_output=embedding_size,
        d_model=embedding_size,
        n_layers=n_layers_s4,
        dropout=0.3,
        prenorm=False)

    
    def forward(self, x):
        if type(x) is list:
            x = torch.cat((x[0], x[1]), dim=1)
        x = x.to(self.device)
        mask = None
        if self.is_mask:
            mask, masked_input = self.mask(x)
        else: 
            masked_input = x

        conv_output = self.conv_encoder(masked_input.clone())
        conv_output = self.conv_encoder2(conv_output)

        if self.use_s4:
            mid_output = self.s4_model(conv_output.transpose(-1,-2).clone())
            mid_output = mid_output.transpose(1,2)
        else:
            mid_output = conv_output # not s4 output
        

        decoder_out = self.deconv_encoder2(mid_output.clone())
        decoder_out = self.deconv_encoder(decoder_out)

        return x, mask, mid_output, decoder_out[:,:,:x.size(2)]

