import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import *
import typing
from typing import Tuple
from sizes import *

if typing.TYPE_CHECKING:
    from .game import Game


class Brain:
    def __init__ (self, game: 'Game') -> None:
        self.game: 'Game' = game

        # (batch, depth, x, y)
        self.input, current = self.stem ()  # (48, 35, 35)

        for i in range (3):
            current = self.inception_a (current, i)  # (48, 35, 35)
        current = self.reduce_a_to_b (current)  # (96, 17, 17)
        for i in range (3):
            current = self.inception_b (current, i)  # (96, 17, 17)
        current = self.reduce_b_to_c (current)  # (144, 8, 8)
        for i in range (3):
            current = self.inception_c (current, i)  # (144, 8, 8)

        final_avg_pool = AveragePooling2D ((8, 8), 1, 'valid', name = 'final_avg_pool') (current)  # (144, 1, 1)
        final_fully_connected = Dense (4, name = 'final_fully_connected') (final_avg_pool)
        self.output = final_fully_connected
        self.net = keras.Model (inputs = self.input, outputs = self.output)
        self.net.compile (optimizer = keras.optimizers.Adam (lr = 0.01))

    # Based on Inception v4 but smaller in scale

    @classmethod
    def stem (cls) -> Tuple[Layer, Layer]:
        """Prepare inputs for further processing. Fit spatial dimension, increase channels."""
        input = layers.Input (name = 'input', shape = (2, 41, 41))  # (2, 41, 41)
        stem_conv1 = Conv2D (8, (3, 3), 1, 'valid', activation = 'relu', name = 'stem_conv1') (input)  # (8, 39, 39)
        stem_conv2 = Conv2D (16, (3, 3), 1, 'valid', activation = 'relu', name = 'stem_conv2') (stem_conv1)  # (16, 37, 37)
        stem_max_pool = MaxPooling2D ((3, 3), 1, 'valid', name = 'stem_max_pool') (stem_conv2)  # (16, 35, 35)
        stem_conv3 = Conv2D (32, (3, 3), 1, 'valid', activation = 'relu', name = 'stem_conv3') (stem_conv2)  # (32, 35, 35)
        stem_output = Concatenate (axis = 1) ([stem_max_pool, stem_conv3])  # (48, 35, 35)
        return input, stem_output

    @classmethod
    def inception_a (cls, input: Layer, index: int) -> Layer:
        """Process at as close level as possible."""
        prefix = f'inception_a_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (48, 35, 35)
        line1_output = Conv2D (12, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line1') (line1_avg_pool)  # (12, 35, 35)
        line2_output = Conv2D (12, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line2') (input)  # (12, 35, 35)
        line3_conv1 = Conv2D (8, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv1') (input)  # (8, 35, 35)
        line3_output = Conv2D (12, (3, 3), 1, 'same', activation = 'relu', name = f'{prefix}_line3') (line3_conv1)  # (12, 35, 35)
        line4_output = Conv2D (12, (3, 3), 1, 'same', activation = 'relu', name = f'{prefix}_line4') (line3_output)  # (12, 35, 35)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (48, 35, 35)
        return output

    @classmethod
    def reduce_a_to_b (cls, input: Layer) -> Layer:
        prefix = 'reduce_a_to_b'
        line1_output = MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (48, 17, 17)
        line2_putput = Conv2D (24, (3, 3), 2, 'valid', activation = 'relu', name = f'{prefix}_line2') (input)  # (24, 17, 17)
        line3_conv1 = Conv2D (12, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv1') (input)  # (12, 35, 35)
        line3_conv2 = Conv2D (18, (3, 3), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv2') (line3_conv1)  # (18, 35, 35)
        line3_output = Conv2D (24, (3, 3), 2, 'valid', activation = 'relu', name = f'{prefix}_line3_conv3') (line3_conv2)  # (24, 17, 17)
        output = Concatenate (axis = 1) ([line1_output, line2_putput, line3_output])  # (96, 17, 17)
        return output

    @classmethod
    def inception_b (cls, input: Layer, index: int) -> Layer:
        """Process at middle level."""
        prefix = f'inception_b_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (96, 17, 17)
        line1_output = Conv2D (24, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line1') (line1_avg_pool)  # (24, 17, 17)
        line2_output = Conv2D (24, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line2') (input)  # (24, 17, 17)
        line3_conv1 = Conv2D (12, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv1') (input)  # (12, 17, 17)
        line3_conv2 = Conv2D (18, (1, 7), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv2') (line3_conv1)  # (18, 17, 17)
        line3_output = Conv2D (24, (7, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3') (line3_conv2)  # (24, 17, 17)
        line4_conv1 = Conv2D (12, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv1') (input)  # (12, 17, 17)
        line4_conv2 = Conv2D (12, (1, 7), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv2') (line4_conv1)  # (12, 17, 17)
        line4_conv3 = Conv2D (18, (7, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv3') (line4_conv2)  # (18, 17, 17)
        line4_conv4 = Conv2D (18, (1, 7), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv3') (line4_conv3)  # (18, 17, 17)
        line4_output = Conv2D (24, (7, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line4') (line4_conv4)  # (24, 17, 17)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (96, 17, 17)
        return output

    @classmethod
    def reduce_b_to_c (cls, input: Layer) -> Layer:
        prefix = 'reduce_b_to_c'
        line1_output = MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (96, 8, 8)
        line2_conv1 = Conv2D (12, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line2_conv1') (input)  # (12, 17, 17)
        line2_output = Conv2D (12, (3, 3), 2, 'valid', activation = 'relu', name = f'{prefix}_line2') (line2_conv1)  # (12, 8, 8)
        line3_conv1 = Conv2D (24, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv1') (input)  # (24, 17, 17)
        line3_conv2 = Conv2D (24, (1, 7), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv2') (line3_conv1)  # (24, 17, 17)
        line3_conv3 = Conv2D (36, (7, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv2') (line3_conv2)  # (36, 17, 17)
        line3_output = Conv2D (36, (3, 3), 2, 'valid', activation = 'relu', name = f'{prefix}_line3_conv3') (line3_conv3)  # (36, 8, 8)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output])  # (144, 8, 8)
        return output

    @classmethod
    def inception_c (cls, input: Layer, index: int) -> Layer:
        """Process at far away look."""
        prefix = f'inception_b_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (144, 8, 8)
        line1_output = Conv2D (24, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line1') (line1_avg_pool)  # (24, 8, 8)
        line2_output = Conv2D (24, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line2') (input)  # (24, 8, 8)
        line3_conv1 = Conv2D (36, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_conv1') (input)  # (36, 8, 8)
        line3_fork_a = Conv2D (24, (1, 3), 1, 'same', activation = 'relu', name = f'{prefix}_line3_fork_a') (line3_conv1)  # (18, 8, 8)
        line3_fork_b = Conv2D (24, (3, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line3_fork_b') (line3_conv1)  # (24, 8, 8)
        line4_conv1 = Conv2D (36, (1, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv1') (input)  # (36, 8, 8)
        line4_conv2 = Conv2D (48, (1, 3), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv2') (line4_conv1)  # (48, 8, 8)
        line4_conv3 = Conv2D (60, (3, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line4_conv3') (line4_conv2)  # (60, 8, 8)
        line4_fork_a = Conv2D (24, (1, 3), 1, 'same', activation = 'relu', name = f'{prefix}_line4_fork_a') (line4_conv3)  # (24, 8, 8)
        line4_fork_b = Conv2D (24, (3, 1), 1, 'same', activation = 'relu', name = f'{prefix}_line4_fork_b') (line4_conv3)  # (24, 8, 8)
        output = Concatenate (axis = 1) ([
            line1_output, line2_output, line3_fork_a, line3_fork_b, line4_fork_a, line4_fork_b])  # (144, 8, 8)
        return output
