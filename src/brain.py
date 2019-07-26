import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kreg
import typing
from typing import Tuple
from sizes import *
import random
import numpy as np
from tkinter import Canvas

from const import *
from iterators import first
from color import tkinter_rgb
from pos import Pos
from dir import Dir
from board import Board
from memories import Memory, Memories

if typing.TYPE_CHECKING:
    from .game import Game


class Brain:
    BATCH = 40
    ACTIONS = 4
    EXPLORATION_CHANCE = 0.03
    FUTURE_DISCOUNT = 0.95
    LEARNING_RATE = 0.01
    REGULARIZER = 1e-5

    FUTURE_STEPS = 10

    def __init__ (self, game: 'Game') -> None:
        self.game: 'Game' = game

        print (tf.__version__)

        # (batch, depth, y, x)
        with tf.name_scope ('brain'):
            self.sight_input = kl.Input (
                shape = (SIGHT_CHANNELS, SIGHT_DIAMETER, SIGHT_DIAMETER), name = 'sight_input')
            current = self.tiny_brain (self.sight_input)
        action_values = current
        self.estimate_actions = tf.keras.Model (inputs = self.sight_input, outputs = action_values)
        self.estimate_actions.summary ()

        layer: kl.Layer
        first_layer = first (first (layer.variables)
                             for layer in self.estimate_actions.layers
                             if isinstance (layer, kl.Conv2D))
        last_layer = [layer
                      for layer in self.estimate_actions.layers
                      if isinstance (layer, kl.Conv2D)][-1]
        self.observed_layers: List[kl.Layer] = \
            [layer
             for layer in self.estimate_actions.layers
             if isinstance (layer, kl.BatchNormalization)] + [last_layer]
        observed_outputs: List[tf.Tensor] = [layer.output for layer in self.observed_layers]
        self.observe_activations = tf.keras.Model (inputs = self.sight_input, outputs = observed_outputs)

        self.mask = kl.Input (shape = (self.ACTIONS, 1, 1))
        self.total_future_rewards = kl.Input (shape = (1, 1, 1))

        single_action_value = tf.reduce_sum (action_values * self.mask, axis = 1, keepdims = True)

        q_delta = self.total_future_rewards - single_action_value
        q_loss = tf.reduce_mean (tf.square (q_delta) / 2)
        # regularization_loss = tf.add_n (self.estimate_actions.losses)
        total_loss = q_loss # + regularization_loss

        optimizer = tf.train.AdamOptimizer (learning_rate = self.LEARNING_RATE)
        self.optimize_single_step = optimizer.minimize (total_loss)

        with tf.name_scope ('visual_brain'):
            self.sight_var: tf.Variable = tf.get_variable (
                shape = (1, SIGHT_CHANNELS, SIGHT_DIAMETER, SIGHT_DIAMETER), name = 'sight_var',
                initializer = tf.random_normal_initializer (stddev = 0.01))
            self.sight_value = tf.placeholder (tf.float32, shape = (1, SIGHT_CHANNELS, SIGHT_DIAMETER, SIGHT_DIAMETER))
            sight_noise = tf.random_normal (shape = (1, SIGHT_CHANNELS, SIGHT_DIAMETER, SIGHT_DIAMETER), stddev = 0.01)
            self.set_sight_var = tf.assign (self.sight_var, self.sight_value + sight_noise)

            sight_var_input = kl.Input (shape = (SIGHT_CHANNELS, SIGHT_DIAMETER, SIGHT_DIAMETER), tensor = self.sight_var)
            visual_brain = self.tiny_brain (sight_var_input)
            self.visual_brain = tf.keras.Model (inputs = sight_var_input, outputs = visual_brain)

        self.copy_weights = [tf.assign (copy, orig) for (copy, orig) in
                             zip (tf.trainable_variables ('visual_brain'), tf.trainable_variables ('brain'))]

        self.session = tf.Session ()
        self.session.run (tf.global_variables_initializer ())

        self.current_sight = None
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
        if len (memories.items) < self.BATCH + self.FUTURE_STEPS:
            return
        batch = random.sample (memories.items[:-self.FUTURE_STEPS], self.BATCH)

        total_future_rewards = self.estimate_future (batch)  # (BATCH, 1, 1, 1)

        prev_sight = Board.make_buffer (self.BATCH)
        mask = self.zero_mask ()
        memory: Memory
        for i, memory in enumerate (batch):
            memory.prev_board.observe (prev_sight[i:i + 1, :, :, :])
            mask[i, memory.action_index, 0, 0] = 1

        self.optimize_single_step.run (feed_dict = {
            self.sight_input: prev_sight,
            self.mask: mask,
            self.total_future_rewards: total_future_rewards,
        })

    def estimate_future (self, batch: List[Memory]) -> np.ndarray:
        tracks: List[List[Memory]] = [[memory] for memory in batch]
        still_alive: List[bool] = []
        rewards = [memory.reward for memory in batch]
        future_sight = Board.make_buffer (self.BATCH)
        for i in range (self.BATCH):
            for step in range (self.FUTURE_STEPS):
                step_discount = self.FUTURE_DISCOUNT ** (step + 1)
                if not tracks[i][step].is_alive:
                    still_alive.append (False)
                    break
                if step == self.FUTURE_STEPS - 1:
                    still_alive.append (True)
                    tracks[i][step].next_board.observe (future_sight[i:i + 1, :, :, :])
                    break
                next_memory = self.game.memories.items[tracks[i][step].index + 1]
                tracks[i].append (next_memory)
                rewards[i] += next_memory.reward * step_discount

        last_day_values = self.estimate_actions.predict (future_sight)  # (BATCH, ACTIONS, 1, 1)

        last_step_discount = self.FUTURE_DISCOUNT ** self.FUTURE_STEPS
        for i in range (self.BATCH):
            if still_alive[i]:
                rewards[i] += last_step_discount * float (np.max (last_day_values[i, :, 0, 0]))
        return np.reshape (rewards, (self.BATCH, 1, 1, 1))

    @classmethod
    def zero_mask (cls) -> np.ndarray:
        return np.zeros ((cls.BATCH, cls.ACTIONS, 1, 1))

    def remember_predictions (self) -> None:
        sight = Board.make_buffer (batch = 1)
        self.game.board.observe (sight)
        self.current_sight = sight
        self.predicted_action_values: np.ndarray = self.estimate_actions.predict (sight).flatten ()
        if self.game.displays:
            self.game.displays.update_images (sight)

    def display_activations (self, sight: np.ndarray) -> List[np.ndarray]:
        return self.observe_activations.predict (sight)
        # return self.session.run (self.observed_outputs, feed_dict = { self.sight_input: sight_input })


    def tiny_brain (self, input: kl.Layer) -> kl.Layer:
        current = input
        current = self.conv2d_bn (current, 16, (3, 3), 2, 'valid', name = 'conv1')  # (16, 20, 20)
        current = self.conv2d_bn (current, 24, (3, 3), 1, 'valid', name = 'conv2')  # (24, 18, 18)
        current = kl.AveragePooling2D ((2, 2), 2, 'valid', name = 'avg_pool') (current)  # (24, 9, 9)
        current = self.conv2d_bn (current, 32, (3, 3), 1, 'valid', name = 'conv3')  # (32, 7, 7)
        current = self.conv2d_bn (current, 64, (3, 3), 2, 'valid', name = 'conv4')  # (64, 3, 3)
        current = kl.AveragePooling2D ((3, 3), 1, 'valid', name = 'final_avg_pool') (current)  # (64, 1, 1)
        current = kl.Conv2D (
            self.ACTIONS, (1, 1), 1, 'valid', name = 'final_fully_connected') (current)  # (4, 1, 1)
        return current

    # Based on Inception v4 but smaller in scale

    def inception_brain (self, input: kl.Layer) -> kl.Layer:
        current = self.stem (input)  # (48, 35, 35)

        for i in range (1):
            current = self.inception_a (current, i)  # (48, 35, 35)
        current = self.reduce_a_to_b (current)  # (96, 17, 17)
        for i in range (1):
            current = self.inception_b (current, i)  # (96, 17, 17)
        current = self.reduce_b_to_c (current)  # (144, 8, 8)
        for i in range (1):
            current = self.inception_c (current, i)  # (144, 8, 8)

        current = kl.AveragePooling2D ((8, 8), 1, 'valid', name = 'final_avg_pool') (current)  # (144, 1, 1)
        current = kl.Conv2D (
            self.ACTIONS, (1, 1), 1, 'valid', name = 'final_fully_connected') (current)  # (4, 1, 1)
        return current

    @classmethod
    def conv2d_bn (cls, current: kl.Layer,
                   filters: int, filter_size: Tuple[int, int], stride: int, padding: str, name: str) -> kl.Layer:
        current = kl.Conv2D (
            filters, filter_size, stride, padding, name = name,
            # kernel_regularizer = kreg.l2 (cls.REGULARIZER), bias_regularizer = kreg.l2 (cls.REGULARIZER)
        ) (current)
        current = kl.BatchNormalization (
            axis = 1, name = f'{name}_bn', trainable = True,
            # beta_regularizer = kreg.l2 (cls.REGULARIZER), gamma_regularizer = kreg.l2 (cls.REGULARIZER)
        ) (current)
        current = kl.Activation ('relu', name = f'{name}_relu') (current)
        return current

    def stem (self, input: kl.Layer) -> kl.Layer:
        """Prepare inputs for further processing. Fit spatial dimension, increase channels."""
        stem_conv1 = self.conv2d_bn (input, 8, (3, 3), 1, 'valid', name = 'stem_conv1')  # (8, 39, 39)
        stem_conv2 = self.conv2d_bn (stem_conv1, 16, (3, 3), 1, 'valid', name = 'stem_conv2')  # (16, 37, 37)
        stem_max_pool = kl.MaxPooling2D ((3, 3), 1, 'valid', name = 'stem_max_pool') (stem_conv2)  # (16, 35, 35)
        stem_conv3 = self.conv2d_bn (stem_conv2, 32, (3, 3), 1, 'valid', name = 'stem_conv3')  # (32, 35, 35)
        stem_output = kl.Concatenate (axis = 1) ([stem_max_pool, stem_conv3])  # (48, 35, 35)
        return stem_output

    @classmethod
    def inception_a (cls, input: kl.Layer, index: int) -> kl.Layer:
        """Process at as close level as possible."""
        prefix = f'inception_a_{index}'
        line1_avg_pool = kl.AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (48, 35, 35)
        line1_output = cls.conv2d_bn (line1_avg_pool, 12, (1, 1), 1, 'same', name = f'{prefix}_line1')  # (12, 35, 35)
        line2_output = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line2')  # (12, 35, 35)
        line3_conv1 = cls.conv2d_bn (input, 8, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (8, 35, 35)
        line3_output = cls.conv2d_bn (line3_conv1, 12, (3, 3), 1, 'same', name = f'{prefix}_line3')  # (12, 35, 35)
        line4_output = cls.conv2d_bn (line3_output, 12, (3, 3), 1, 'same', name = f'{prefix}_line4')  # (12, 35, 35)
        output = kl.Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (48, 35, 35)
        return output

    @classmethod
    def reduce_a_to_b (cls, input: kl.Layer) -> kl.Layer:
        prefix = 'reduce_a_to_b'
        line1_output = kl.MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (48, 17, 17)
        line2_putput = cls.conv2d_bn (input, 24, (3, 3), 2, 'valid', name = f'{prefix}_line2')  # (24, 17, 17)
        line3_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (12, 35, 35)
        line3_conv2 = cls.conv2d_bn (line3_conv1, 18, (3, 3), 1, 'same', name = f'{prefix}_line3_conv2')  # (18, 35, 35)
        line3_output = cls.conv2d_bn (line3_conv2, 24, (3, 3), 2, 'valid', name = f'{prefix}_line3_conv3')  # (24, 17, 17)
        output = kl.Concatenate (axis = 1) ([line1_output, line2_putput, line3_output])  # (96, 17, 17)
        return output

    @classmethod
    def inception_b (cls, input: kl.Layer, index: int) -> kl.Layer:
        """Process at middle level."""
        prefix = f'inception_b_{index}'
        line1_avg_pool = kl.AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (96, 17, 17)
        line1_output = cls.conv2d_bn (line1_avg_pool, 24, (1, 1), 1, 'same', name = f'{prefix}_line1')  # (24, 17, 17)
        line2_output = cls.conv2d_bn (input, 24, (1, 1), 1, 'same', name = f'{prefix}_line2')  # (24, 17, 17)
        line3_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (12, 17, 17)
        line3_conv2 = cls.conv2d_bn (line3_conv1,18, (1, 7), 1, 'same', name = f'{prefix}_line3_conv2')  # (18, 17, 17)
        line3_output = cls.conv2d_bn (line3_conv2, 24, (7, 1), 1, 'same', name = f'{prefix}_line3')  # (24, 17, 17)
        line4_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line4_conv1')  # (12, 17, 17)
        line4_conv2 = cls.conv2d_bn (line4_conv1, 12, (1, 7), 1, 'same', name = f'{prefix}_line4_conv2')  # (12, 17, 17)
        line4_conv3 = cls.conv2d_bn (line4_conv2, 18, (7, 1), 1, 'same', name = f'{prefix}_line4_conv3')  # (18, 17, 17)
        line4_conv4 = cls.conv2d_bn (line4_conv3, 18, (1, 7), 1, 'same', name = f'{prefix}_line4_conv4')  # (18, 17, 17)
        line4_output = cls.conv2d_bn (line4_conv4, 24, (7, 1), 1, 'same', name = f'{prefix}_line4')  # (24, 17, 17)
        output = kl.Concatenate (axis = 1) ([line1_output, line2_output, line3_output, line4_output])  # (96, 17, 17)
        return output

    @classmethod
    def reduce_b_to_c (cls, input: kl.Layer) -> kl.Layer:
        prefix = 'reduce_b_to_c'
        line1_output = kl.MaxPooling2D ((3, 3), 2, 'valid', name = f'{prefix}_max_pool') (input)  # (96, 8, 8)
        line2_conv1 = cls.conv2d_bn (input, 12, (1, 1), 1, 'same', name = f'{prefix}_line2_conv1')  # (12, 17, 17)
        line2_output = cls.conv2d_bn (line2_conv1, 12, (3, 3), 2, 'valid', name = f'{prefix}_line2')  # (12, 8, 8)
        line3_conv1 = cls.conv2d_bn (input, 24, (1, 1), 1, 'same', name = f'{prefix}_line3_conv1')  # (24, 17, 17)
        line3_conv2 = cls.conv2d_bn (line3_conv1, 24, (1, 7), 1, 'same', name = f'{prefix}_line3_conv2')  # (24, 17, 17)
        line3_conv3 = cls.conv2d_bn (line3_conv2, 36, (7, 1), 1, 'same', name = f'{prefix}_line3_conv3')  # (36, 17, 17)
        line3_output = cls.conv2d_bn (line3_conv3, 36, (3, 3), 2, 'valid', name = f'{prefix}_line3')  # (36, 8, 8)
        output = kl.Concatenate (axis = 1) ([line1_output, line2_output, line3_output])  # (144, 8, 8)
        return output

    @classmethod
    def inception_c (cls, input: kl.Layer, index: int) -> kl.Layer:
        """Process at far away look."""
        prefix = f'inception_c_{index}'
        line1_avg_pool = kl.AveragePooling2D ((3, 3), 1, 'same', name = f'{prefix}_line1_avg_pool') (input)  # (144, 8, 8)
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
        output = kl.Concatenate (axis = 1) ([
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
            color = tkinter_rgb (-2 * value - 1, 2 * value + 1, 1 - 2 * abs (value + 0.5))
            (head + dir.offset).draw (canvas, head, color)
