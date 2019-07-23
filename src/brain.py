import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import typing
from typing import Tuple
from sizes import *
import random
import numpy as np
from tkinter import Canvas

from pos import Pos
from dir import Dir
from board import Board
from memories import Memory, Memories

if typing.TYPE_CHECKING:
    from .game import Game


class Brain:
    BATCH = 50
    ACTIONS = 4
    EXPLORATION_CHANCE = 0.05
    FUTURE_DISCOUNT = 0.95
    LEARNING_RATE = 0.01
    REGULARIZER = 0.01

    FUTURE_STEPS = 1

    def __init__ (self, game: 'Game') -> None:
        self.game: 'Game' = game

        # (batch, depth, y, x)
        self.sight, current = self.tiny_brain ()
        action_values = current
        self.estimate_actions = keras.Model (inputs = self.sight, outputs = action_values)
        self.estimate_actions.summary ()

        self.mask = Input (shape = (self.ACTIONS, 1, 1))
        self.total_future_rewards = Input (shape = (1, 1, 1))

        single_action_value = tf.reduce_sum (action_values * self.mask, axis = 1, keepdims = True)
        self.best_action_value = tf.reduce_max (action_values, axis = 1, keepdims = True)

        q_delta = self.total_future_rewards - single_action_value
        q_loss = tf.reduce_mean (tf.square (q_delta) / 2)
        regularization_loss = tf.add_n (self.estimate_actions.losses)
        total_loss = q_loss + regularization_loss

        optimizer = tf.train.AdamOptimizer (learning_rate = self.LEARNING_RATE)
        self.optimize_single_step = optimizer.minimize (total_loss)

        self.session = tf.Session ()
        self.session.run (tf.global_variables_initializer ())

        self.predicted_action_values = None
        self.last_action_taken = None
        self.remember_predictions ()

    def think (self) -> Tuple[int, bool]:
        turned = False
        action_values = self.predicted_action_values
        print ([f'{Dir.ALL[i]}{action_values[i]:.02f}' for i in range (self.ACTIONS)])
        if random.random () < self.EXPLORATION_CHANCE:
            action_index = random.randrange (len (Dir.ALL))
            print ('Exploring')
        else:
            action_index: int = self.argmax (action_values)
            if self.last_action_taken is not None and action_index != self.last_action_taken:
                if Dir.ALL[action_index].is_opposite (Dir.ALL[self.last_action_taken]):
                    print ('Suicide')
                else:
                    print ('Turn')
                    turned = True
            self.last_action_taken: int = action_index
        return action_index, turned

    def learn (self, memories: Memories) -> None:
        if len (memories.items) < self.BATCH:
            return
        batch = [memories.items[-1]] + random.sample (memories.items, self.BATCH - 1)

        total_future_rewards = self.estimate_future (batch)  # (BATCH, 1, 1, 1)

        prev_sight = Board.make_buffer (self.BATCH)
        mask = self.zero_mask ()
        memory: Memory
        for i, memory in enumerate (batch):
            memory.prev_board.observe (prev_sight[i:i + 1, :, :, :])
            mask[i, memory.action_index, 0, 0] = 1

        self.optimize_single_step.run (feed_dict = {
            self.sight: prev_sight,
            self.mask: mask,
            self.total_future_rewards: total_future_rewards,
        })
        self.remember_predictions ()

    def estimate_future (self, batch: List[Memory]) -> np.ndarray:
        tracks: List[List[Memory]] = [[memory] for memory in batch]
        rewards = [memory.reward for memory in batch]
        for step in range (self.FUTURE_STEPS):
            step_discount = self.FUTURE_DISCOUNT ** (step + 1)
            future_sight = Board.make_buffer (self.BATCH)
            memory: Memory
            for i in range (self.BATCH):
                if step >= len (tracks[i]) or not tracks[i][step].is_alive:
                    continue
                tracks[i][step].next_board.observe (future_sight[i:i + 1, :, :, :])
            tomorrow_values = self.estimate_actions.predict (future_sight)  # (BATCH, ACTIONS, 1, 1)

            for i in range (self.BATCH):
                if step >= len (tracks[i]) or not tracks[i][step].is_alive:
                    continue
                if step == self.FUTURE_STEPS - 1:
                    rewards[i] += step_discount * float (np.max (tomorrow_values[i, :, 0, 0]))
                    continue
                today_board = tracks[i][step].next_board
                action_index = self.argmax (tomorrow_values[i, :, 0, 0])
                tomorrow_board, reward, is_alive = today_board.step (action_index, real_snake = False)
                tracks[i].append (Memory (today_board, action_index, reward, is_alive, tomorrow_board))
                rewards[i] += step_discount * reward
        return np.reshape (rewards, (self.BATCH, 1, 1, 1))

    @classmethod
    def zero_mask (cls) -> np.ndarray:
        return np.zeros ((cls.BATCH, cls.ACTIONS, 1, 1))

    def remember_predictions (self) -> None:
        sight = Board.make_buffer (batch = 1)
        self.game.board.observe (sight)
        self.predicted_action_values: np.ndarray = self.estimate_actions.predict (sight).flatten ()


    def tiny_brain (self) -> Tuple[Layer, Layer]:
        current = Input (shape = (2, 41, 41), name = 'sight')
        sight = current
        current = self.conv2d_bn (current, 16, (3, 3), 2, 'valid', name = 'conv1')  # (16, 20, 20)
        current = self.conv2d_bn (current, 20, (3, 3), 1, 'valid', name = 'conv2')  # (20, 18, 18)
        current = MaxPooling2D ((2, 2), 2, 'valid', name = 'avg_pool') (current)  # (20, 9, 9)
        current = self.conv2d_bn (current, 24, (3, 3), 1, 'valid', name = 'conv3')  # (24, 7, 7)
        current = AveragePooling2D ((7, 7), 1, 'valid', name = 'final_avg_pool') (current)  # (24, 1, 1)
        current = Conv2D (
            self.ACTIONS, (1, 1), 1, 'valid', name = 'final_fully_connected') (current)  # (4, 1, 1)
        return sight, current

    # Based on Inception v4 but smaller in scale

    def inception_brain (self) -> Tuple[Layer, Layer]:
        sight, current = self.stem ()  # (48, 35, 35)

        for i in range (1):
            current = self.inception_a (current, i)  # (48, 35, 35)
        current = self.reduce_a_to_b (current)  # (96, 17, 17)
        for i in range (1):
            current = self.inception_b (current, i)  # (96, 17, 17)
        current = self.reduce_b_to_c (current)  # (144, 8, 8)
        for i in range (1):
            current = self.inception_c (current, i)  # (144, 8, 8)

        current = AveragePooling2D ((8, 8), 1, 'valid', name = 'final_avg_pool') (current)  # (144, 1, 1)
        current = Conv2D (
            self.ACTIONS, (1, 1), 1, 'valid', name = 'final_fully_connected') (current)  # (4, 1, 1)
        return sight, current

    @classmethod
    def conv2d_bn (cls, current: Layer, filters: int, filter_size: Tuple[int, int], stride: int, padding: str, name: str) -> Layer:
        current = Conv2D (
            filters, filter_size, stride, padding, name = name,
            kernel_regularizer = l2 (cls.REGULARIZER), bias_regularizer = l2 (cls.REGULARIZER)
        ) (current)
        current = BatchNormalization (axis = 1, name = f'{name}_bn', trainable = True) (current)
        current = Activation ('relu', name = f'{name}_relu') (current)
        return current

    @classmethod
    def stem (cls) -> Tuple[Layer, Layer]:
        """Prepare inputs for further processing. Fit spatial dimension, increase channels."""
        sight = layers.Input (name = 'sight', shape = (2, 41, 41))  # (2, 41, 41)
        stem_conv1 = cls.conv2d_bn (sight, 8, (3, 3), 1, 'valid', name = 'stem_conv1')  # (8, 39, 39)
        stem_conv2 = cls.conv2d_bn (stem_conv1, 16, (3, 3), 1, 'valid', name = 'stem_conv2')  # (16, 37, 37)
        stem_max_pool = MaxPooling2D (stem_conv2, (3, 3), 1, 'valid', name = 'stem_max_pool')  # (16, 35, 35)
        stem_conv3 = cls.conv2d_bn (stem_conv2, 32, (3, 3), 1, 'valid', name = 'stem_conv3')  # (32, 35, 35)
        stem_output = Concatenate (axis = 1) ([stem_max_pool, stem_conv3])  # (48, 35, 35)
        return sight, stem_output

    @classmethod
    def inception_a (cls, input: Layer, index: int) -> Layer:
        """Process at as close level as possible."""
        prefix = f'inception_a_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (48, 35, 35)
        line1_output = cls.conv2d_bn (line1_avg_pool, 12, (1, 1), 1, 'same', name = f'{prefix}_line1')  # (12, 35, 35)
        line2_output = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line2')  # (12, 35, 35)
        line3_conv1 = cls.conv2d_bn (input, 8, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (8, 35, 35)
        line3_output = cls.conv2d_bn (line3_conv1, 12, (3, 3), 1, 'same', name = f'{prefix}_line3')  # (12, 35, 35)
        line4_output = cls.conv2d_bn (line3_output, 12, (3, 3), 1, 'same', name = f'{prefix}_line4')  # (12, 35, 35)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (48, 35, 35)
        return output

    @classmethod
    def reduce_a_to_b (cls, input: Layer) -> Layer:
        prefix = 'reduce_a_to_b'
        line1_output = MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (48, 17, 17)
        line2_putput = cls.conv2d_bn (input, 24, (3, 3), 2, 'valid', name = f'{prefix}_line2')  # (24, 17, 17)
        line3_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (12, 35, 35)
        line3_conv2 = cls.conv2d_bn (line3_conv1, 18, (3, 3), 1, 'same', name = f'{prefix}_line3_conv2')  # (18, 35, 35)
        line3_output = cls.conv2d_bn (line3_conv2, 24, (3, 3), 2, 'valid', name = f'{prefix}_line3_conv3')  # (24, 17, 17)
        output = Concatenate (axis = 1) ([line1_output, line2_putput, line3_output])  # (96, 17, 17)
        return output

    @classmethod
    def inception_b (cls, input: Layer, index: int) -> Layer:
        """Process at middle level."""
        prefix = f'inception_b_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (96, 17, 17)
        line1_output = cls.conv2d_bn (line1_avg_pool, 24, (1, 1), 1, 'same', name = f'{prefix}_line1')  # (24, 17, 17)
        line2_output = cls.conv2d_bn (input, 24, (1, 1), 1, 'same', name = f'{prefix}_line2')  # (24, 17, 17)
        line3_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (12, 17, 17)
        line3_conv2 = cls.conv2d_bn (line3_conv1,18, (1, 7), 1, 'same', name = f'{prefix}_line3_conv2')  # (18, 17, 17)
        line3_output = cls.conv2d_bn (line3_conv2, 24, (7, 1), 1, 'same', name = f'{prefix}_line3')  # (24, 17, 17)
        line4_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line4_conv1')  # (12, 17, 17)
        line4_conv2 = cls.conv2d_bn (line4_conv1, 12, (1, 7), 1, 'same', name = f'{prefix}_line4_conv2')  # (12, 17, 17)
        line4_conv3 = cls.conv2d_bn (line4_conv2, 18, (7, 1), 1, 'same', name = f'{prefix}_line4_conv3')  # (18, 17, 17)
        line4_conv4 = cls.conv2d_bn (line4_conv318, (1, 7), 1, 'same', name = f'{prefix}_line4_conv4')  # (18, 17, 17)
        line4_output = cls.conv2d_bn (line4_conv4, 24, (7, 1), 1, 'same', name = f'{prefix}_line4')  # (24, 17, 17)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (96, 17, 17)
        return output

    @classmethod
    def reduce_b_to_c (cls, input: Layer) -> Layer:
        prefix = 'reduce_b_to_c'
        line1_output = MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (96, 8, 8)
        line2_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line2_conv1')  # (12, 17, 17)
        line2_output = cls.conv2d_bn (line2_conv1, 12, (3, 3), 2, 'valid', name = f'{prefix}_line2')  # (12, 8, 8)
        line3_conv1 = cls.conv2d_bn (input, 24, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (24, 17, 17)
        line3_conv2 = cls.conv2d_bn (line3_conv1, 24, (1, 7), 1, 'same', name = f'{prefix}_line3_conv2')  # (24, 17, 17)
        line3_conv3 = cls.conv2d_bn (line3_conv2, 36, (7, 1), 1, 'same', name = f'{prefix}_line3_conv3')  # (36, 17, 17)
        line3_output = cls.conv2d_bn (line3_conv3, 36, (3, 3), 2, 'valid', name = f'{prefix}_line3')  # (36, 8, 8)
        output = Concatenate (axis = 1) ([line1_output, line2_output, line3_output])  # (144, 8, 8)
        return output

    @classmethod
    def inception_c (cls, input: Layer, index: int) -> Layer:
        """Process at far away look."""
        prefix = f'inception_c_{index}'
        line1_avg_pool = AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (144, 8, 8)
        line1_output = cls.conv2d_bn (line1_avg_pool, 24, (1, 1), 1, 'same', name = f'{prefix}_line1')  # (24, 8, 8)
        line2_output = cls.conv2d_bn (input, 24, (1, 1), 1, 'same', name = f'{prefix}_line2')  # (24, 8, 8)
        line3_conv1 = cls.conv2d_bn (input, 36, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (36, 8, 8)
        line3_fork_a = cls.conv2d_bn (line3_conv1, 24, (1, 3), 1, 'same', name = f'{prefix}_line3_fork_a')  # (18, 8, 8)
        line3_fork_b = cls.conv2d_bn (line3_conv1, 24, (3, 1), 1, 'same', name = f'{prefix}_line3_fork_b')  # (24, 8, 8)
        line4_conv1 = cls.conv2d_bn (input, 36, (1, 1), 1, 'same', name = f'{prefix}_line4_conv1')  # (36, 8, 8)
        line4_conv2 = cls.conv2d_bn (line4_conv1, 48, (1, 3), 1, 'same', name = f'{prefix}_line4_conv2')  # (48, 8, 8)
        line4_conv3 = cls.conv2d_bn (line4_conv2, 60, (3, 1), 1, 'same', name = f'{prefix}_line4_conv3')  # (60, 8, 8)
        line4_fork_a = cls.conv2d_bn (line4_conv3, 24, (1, 3), 1, 'same', name = f'{prefix}_line4_fork_a')  # (24, 8, 8)
        line4_fork_b = cls.conv2d_bn (line4_conv3, 24, (3, 1), 1, 'same', name = f'{prefix}_line4_fork_b')  # (24, 8, 8)
        output = Concatenate (axis = 1) ([
            line1_output, line2_output, line3_fork_a, line3_fork_b, line4_fork_a, line4_fork_b])  # (144, 8, 8)
        return output

    def on_death (self) -> None:
        self.last_action_taken = None

    @staticmethod
    def argmax (values) -> int:
        return max (range (len (values)), key = lambda i: values[i])

    def draw (self, canvas: Canvas, head: Pos) -> None:
        for i, dir in enumerate (Dir.ALL):
            value = self.predicted_action_values[i]
            color = self.rgb (-2 * value - 1, 2 * value + 1, 1 - 2 * abs (value + 0.5))
            (head + dir.offset).draw (canvas, head, color)

    @classmethod
    def rgb (cls, r: float, g: float, b: float) -> str:
        r = max (0, min (255, int (round (r * 255))))
        g = max (0, min (255, int (round (g * 255))))
        b = max (0, min (255, int (round (b * 255))))
        return f'#{r:02x}{g:02x}{b:02x}'
