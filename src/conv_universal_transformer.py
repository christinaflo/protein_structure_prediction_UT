"""
Encoder block and transition function definitions for a convolutional-augmented Universal Transformer.
"""

import math
from typing import Union, Callable, Optional
from keras import backend as K
from keras.layers import Dropout, Conv1D, concatenate, SeparableConv1D
from keras.utils import get_custom_objects
from keras_transformer.transformer import TransformerBlock, LayerNormalization


def gelu(x):
    """     
    NOTE: This function is from the included keras-transformer repository.
    There was strange behavior when attemping to use imported function
    (wildly fluctuating accuracy).
    https://github.com/kpot/keras-transformer

    GeLU was used to prevent dead ReLU problem that we encounted a few times.
    """     
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + K.tanh(c * (x + 0.044715 * K.pow(x, 3))))

class ConvTransformerTransition:
    """
    Transition function performed at the end of each encoder block.
    Three 1D convolutions, with 2x, 4x, 1x sized feature maps relative
    to input hidden dimensions. Last convolution is a inear projection
    with no activation.
    """

    def __init__(self, name: str, activation: Union[str, Callable], filter_size: int):

        self.transition_conv1 = Conv1D(name=f'{name}_conv1',
                                       filters=2*filter_size,
                                       strides=1,
                                       padding='same',
                                       activation=activation,
                                       kernel_size=3)
        self.transition_conv2 = Conv1D(name=f'{name}_conv2',
                                       filters=4*filter_size,
                                       strides=1,
                                       padding='same',
                                       activation=activation,
                                       kernel_size=3)
        self.transition_conv3 = Conv1D(name=f'{name}_conv3',
                                       filters=filter_size,
                                       strides=1,
                                       padding='same',
                                       kernel_size=5)
        
        self.norm_layer = LayerNormalization(name=f'{name}_normalization')

    def __call__(self, _input): 
        x = self.norm_layer(self.transition_conv1(_input))
        x = self.transition_conv2(x)
        return self.transition_conv3(x)

class ConvUniversalTransformerBlock(TransformerBlock):
    """
    Subclass of TransformerBlock that replaces the transition function and
    prepends an additional convolutional block.

    The author of the TransformerBlock class mentions that UT adds dropout
    after residual connection instead of before it, as in a traditional transformer.
    This is true in the first revision of the paper but is not the case in the latest
    revision. Regardless, it seems to provide better results for our purposes, so we
    have chosen to use this version as well.

    Order of operations:
        - 1D Convolution (3x3, 128 feature maps)
        - Residual Connection
        - Dropout
        - Layer Normalization

        - Multi-head self-attention (unmasked)
        - Residual Connection
        - Dropout
        - Layer Normalization

        - Transition Function
        - Residual Connection
        - Dropout
        - Layer Normalization
    """

    def __init__(self, name: str, num_heads: int, filter_size: int,
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 conv_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'gelu',
                 compression_window_size: int = None,
                 use_masking: bool = False):

        super().__init__(name=name, num_heads=num_heads,
                         residual_dropout=residual_dropout,
                         attention_dropout=attention_dropout,
                         activation=activation,
                         compression_window_size=compression_window_size,
                         use_masking=use_masking)

        self.conv1 = Conv1D(name=f'{name}_conv',
                            filters=filter_size,
                            strides=1,
                            activation=activation,
                            padding='same',
                            kernel_size=3)

        self.conv_dropout_layer = (
            Dropout(conv_dropout)
            if conv_dropout > 0 else lambda x: x)

        self.transition_layer = ConvTransformerTransition(name=f'{name}_transition',
                                                          activation=activation,
                                                          filter_size=filter_size)

        self.norm3_layer = LayerNormalization(name=f'{name}_normalization3')

    def __call__(self, _input):
        conv1 = self.conv1(_input)
        post_residual1 = self.conv_dropout_layer(self.addition_layer([_input, conv1]))
        norm1_output = self.norm1_layer(post_residual1)

        output = self.attention_layer(norm1_output)
        post_residual2 = self.dropout_layer(self.addition_layer([norm1_output, output]))
        norm2_output = self.norm2_layer(post_residual2)

        output = self.transition_layer(norm2_output)
        post_residual3 = self.dropout_layer(self.addition_layer([norm2_output, output]))
        output = self.norm3_layer(post_residual3)
        return output

get_custom_objects().update({
    'ConvUniversalTransformerBlock': ConvUniversalTransformerBlock, 
    'ConvTransformerTransition': ConvTransformerTransition,
    'gelu': gelu,
})

