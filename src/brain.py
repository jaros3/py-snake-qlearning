import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import typing
from typing import Tuple
from sizes import *
import random
from dir import Dir
import numpy as np

if typing.TYPE_CHECKING:
    from .game import Game


class Brain:
    EXPLORATION_CHANCE = 0.05
    FUTURE_DISCOUNT = 0.95
    LEARNING_RATE = 0.01
    REGULARIZER = 0.01

    def __init__ (self, game: 'Game') -> None:
        self.game: 'Game' = game

        # (batch, depth, y, x)
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
        final_fully_connected = Conv2D (4, (1, 1), 1, 'valid', name = 'final_fully_connected') (final_avg_pool)
        self.action_values = final_fully_connected
        self.estimate_actions = keras.Model (inputs = self.input, outputs = self.action_values)
        self.estimate_actions.summary ()

        self.reward: tf.Variable = tf.Variable (0.0, trainable = False)
        self.next_action_value: tf.Variable = tf.Variable (0.0, trainable = False)
        # noinspection PyTypeChecker
        delta_weights = self.reward + self.FUTURE_DISCOUNT * self.next_action_value - self.action_values
        regularization_loss = tf.add_n (self.estimate_actions.losses)
        self.q_learning_loss = tf.square (delta_weights) / 2 + regularization_loss

        optimizer = tf.train.AdamOptimizer (learning_rate = self.LEARNING_RATE)
        self.optimizer_single_step = optimizer.minimize (self.q_learning_loss)

        self.session = tf.Session ()
        self.session.run (tf.global_variables_initializer ())

    # Based on Inception v4 but smaller in scale

    @classmethod
    def conv2d (cls, filters: int, filter_size: Tuple[int, int], stride: int, padding: str, name: str) -> Layer:
        return Conv2D (
            filters, filter_size, stride, padding, name = name, activation = 'relu',
            kernel_regularizer = l2 (cls.REGULARIZER), bias_regularizer = l2 (cls.REGULARIZER)
        )

    @classmethod
    def stem (cls) -> Tuple[Layer, Layer]:
        """Prepare inputs for further processing. Fit spatial dimension, increase channels."""
        input = layers.Input (name = 'input', shape = (2, 41, 41))  # (2, 41, 41)
        stem_conv1 = cls.conv2d (8, (3, 3), 1, 'valid', name = 'stem_conv1') (input)  # (8, 39, 39)
        stem_conv2 = cls.conv2d (16, (3, 3), 1, 'valid', name = 'stem_conv2') (stem_conv1)  # (16, 37, 37)
        stem_max_pool = MaxPooling2D ((3, 3), 1, 'valid', name = 'stem_max_pool') (stem_conv2)  # (16, 35, 35)
        stem_conv3 = cls.conv2d (32, (3, 3), 1, 'valid', name = 'stem_conv3') (stem_conv2)  # (32, 35, 35)
        stem_output = Concatenate (axis = 1) ([stem_max_pool, stem_conv3])  # (48, 35, 35)
        return input, stem_output

    @classmethod
    def inception_a (cls, input: Layer, index: int) -> Layer:
        """Process at as close level as possible."""
        prefix = f'inception_a_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (48, 35, 35)
        line1_output = cls.conv2d (12, (1, 1), 1, 'same', name = f'{prefix}_line1') (line1_avg_pool)  # (12, 35, 35)
        line2_output = cls.conv2d (12, (1, 1), 1, 'same', name = f'{prefix}_line2') (input)  # (12, 35, 35)
        line3_conv1 = cls.conv2d (8, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1') (input)  # (8, 35, 35)
        line3_output = cls.conv2d (12, (3, 3), 1, 'same', name = f'{prefix}_line3') (line3_conv1)  # (12, 35, 35)
        line4_output = cls.conv2d (12, (3, 3), 1, 'same', name = f'{prefix}_line4') (line3_output)  # (12, 35, 35)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (48, 35, 35)
        return output

    @classmethod
    def reduce_a_to_b (cls, input: Layer) -> Layer:
        prefix = 'reduce_a_to_b'
        line1_output = MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (48, 17, 17)
        line2_putput = cls.conv2d (24, (3, 3), 2, 'valid', name = f'{prefix}_line2') (input)  # (24, 17, 17)
        line3_conv1 = cls.conv2d (12, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1') (input)  # (12, 35, 35)
        line3_conv2 = cls.conv2d (18, (3, 3), 1, 'same', name = f'{prefix}_line3_conv2') (line3_conv1)  # (18, 35, 35)
        line3_output = cls.conv2d (24, (3, 3), 2, 'valid', name = f'{prefix}_line3_conv3') (line3_conv2)  # (24, 17, 17)
        output = Concatenate (axis = 1) ([line1_output, line2_putput, line3_output])  # (96, 17, 17)
        return output

    @classmethod
    def inception_b (cls, input: Layer, index: int) -> Layer:
        """Process at middle level."""
        prefix = f'inception_b_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (96, 17, 17)
        line1_output = cls.conv2d (24, (1, 1), 1, 'same', name = f'{prefix}_line1') (line1_avg_pool)  # (24, 17, 17)
        line2_output = cls.conv2d (24, (1, 1), 1, 'same', name = f'{prefix}_line2') (input)  # (24, 17, 17)
        line3_conv1 = cls.conv2d (12, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1') (input)  # (12, 17, 17)
        line3_conv2 = cls.conv2d (18, (1, 7), 1, 'same', name = f'{prefix}_line3_conv2') (line3_conv1)  # (18, 17, 17)
        line3_output = cls.conv2d (24, (7, 1), 1, 'same', name = f'{prefix}_line3') (line3_conv2)  # (24, 17, 17)
        line4_conv1 = cls.conv2d (12, (1, 1), 1, 'same', name = f'{prefix}_line4_conv1') (input)  # (12, 17, 17)
        line4_conv2 = cls.conv2d (12, (1, 7), 1, 'same', name = f'{prefix}_line4_conv2') (line4_conv1)  # (12, 17, 17)
        line4_conv3 = cls.conv2d (18, (7, 1), 1, 'same', name = f'{prefix}_line4_conv3') (line4_conv2)  # (18, 17, 17)
        line4_conv4 = cls.conv2d (18, (1, 7), 1, 'same', name = f'{prefix}_line4_conv4') (line4_conv3)  # (18, 17, 17)
        line4_output = cls.conv2d (24, (7, 1), 1, 'same', name = f'{prefix}_line4') (line4_conv4)  # (24, 17, 17)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (96, 17, 17)
        return output

    @classmethod
    def reduce_b_to_c (cls, input: Layer) -> Layer:
        prefix = 'reduce_b_to_c'
        line1_output = MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (96, 8, 8)
        line2_conv1 = cls.conv2d (12, (1, 1), 1, 'same', name = f'{prefix}_line2_conv1') (input)  # (12, 17, 17)
        line2_output = cls.conv2d (12, (3, 3), 2, 'valid', name = f'{prefix}_line2') (line2_conv1)  # (12, 8, 8)
        line3_conv1 = cls.conv2d (24, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1') (input)  # (24, 17, 17)
        line3_conv2 = cls.conv2d (24, (1, 7), 1, 'same', name = f'{prefix}_line3_conv2') (line3_conv1)  # (24, 17, 17)
        line3_conv3 = cls.conv2d (36, (7, 1), 1, 'same', name = f'{prefix}_line3_conv3') (line3_conv2)  # (36, 17, 17)
        line3_output = cls.conv2d (36, (3, 3), 2, 'valid', name = f'{prefix}_line3') (line3_conv3)  # (36, 8, 8)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output])  # (144, 8, 8)
        return output

    @classmethod
    def inception_c (cls, input: Layer, index: int) -> Layer:
        """Process at far away look."""
        prefix = f'inception_c_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (144, 8, 8)
        line1_output = cls.conv2d (24, (1, 1), 1, 'same', name = f'{prefix}_line1') (line1_avg_pool)  # (24, 8, 8)
        line2_output = cls.conv2d (24, (1, 1), 1, 'same', name = f'{prefix}_line2') (input)  # (24, 8, 8)
        line3_conv1 = cls.conv2d (36, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1') (input)  # (36, 8, 8)
        line3_fork_a = cls.conv2d (24, (1, 3), 1, 'same', name = f'{prefix}_line3_fork_a') (line3_conv1)  # (18, 8, 8)
        line3_fork_b = cls.conv2d (24, (3, 1), 1, 'same', name = f'{prefix}_line3_fork_b') (line3_conv1)  # (24, 8, 8)
        line4_conv1 = cls.conv2d (36, (1, 1), 1, 'same', name = f'{prefix}_line4_conv1') (input)  # (36, 8, 8)
        line4_conv2 = cls.conv2d (48, (1, 3), 1, 'same', name = f'{prefix}_line4_conv2') (line4_conv1)  # (48, 8, 8)
        line4_conv3 = cls.conv2d (60, (3, 1), 1, 'same', name = f'{prefix}_line4_conv3') (line4_conv2)  # (60, 8, 8)
        line4_fork_a = cls.conv2d (24, (1, 3), 1, 'same', name = f'{prefix}_line4_fork_a') (line4_conv3)  # (24, 8, 8)
        line4_fork_b = cls.conv2d (24, (3, 1), 1, 'same', name = f'{prefix}_line4_fork_b') (line4_conv3)  # (24, 8, 8)
        output = Concatenate (axis = 1) ([
            line1_output, line2_output, line3_fork_a, line3_fork_b, line4_fork_a, line4_fork_b])  # (144, 8, 8)
        return output

    def think (self, sight: np.ndarray) -> Dir:
        action_values: np.ndarray = self.estimate_actions.predict (sight).flatten ()
        print (f'{type (action_values)}: {action_values}')
        if random.random () < self.EXPLORATION_CHANCE:
            action_index = random.randrange (len (Dir.ALL))
        else:
            action_index = self.argmax (action_values)
        # action_value = action_values[action_index]
        return Dir.ALL[action_index]

    @staticmethod
    def argmax (values) -> float:
        return max (range (len (values)), key = lambda i: values[i])

    def learn (self, reward: float, last_sight: np.ndarray) -> None:
        next_action_value = np.max (self.estimate_actions.predict (self.game.sight ()))

        #q_learning_loss = self.q_learning_loss.predict (last_sight, feed_dict = {
        #    self.reward: reward, self.next_action_value: next_action_value
        #})
        self.optimizer_single_step.run (feed_dict = {
            self.input: last_sight, self.reward: reward, self.next_action_value: next_action_value
        })
