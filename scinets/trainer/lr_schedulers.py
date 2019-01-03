__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import tensorflow as tf
from .._backend_utils import SubclassRegister
from abc import ABC, abstractmethod


lr_scheduler_register = SubclassRegister("Learning rate scheduler")


def get_lr_scheduler(item):
    return lr_scheduler_register.get_item(item)


@lr_scheduler_register.link_base
class BaseLRScheduler(ABC):
    def __init__(self, learning_rate, global_step, steps_per_epoch, name=None):
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.steps_per_epoch = steps_per_epoch
        self.name = name

    @abstractmethod
    def build_lr_scheduler(self):
        pass


class ExponentialDecay(BaseLRScheduler):
    def __init__(
        self,
        learning_rate,
        global_step,
        steps_per_epoch,
        decay_steps,
        decay_rate,
        staircase=False,
        name=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            global_step=global_step,
            name=name,
            steps_per_epoch=steps_per_epoch,
        )
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def build_lr_scheduler(self):
        return tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase,
            name=self.name,
        )


class PolynomialDecay(BaseLRScheduler):
    def __init__(
        self,
        learning_rate,
        global_step,
        steps_per_epoch,
        decay_steps,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False,
        name=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            global_step=global_step,
            name=name,
            steps_per_epoch=steps_per_epoch,
        )
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        power = power
        self.cycle = cycle

    def build_lr_scheduler(self):
        return tf.train.polynomial_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            end_learning_rate=self.end_learning_rate,
            power=self.power,
            cycle=self.cycle,
            name=self.name,
        )


class CosineDecay(BaseLRScheduler):
    def __init__(
        self,
        learning_rate,
        global_step,
        steps_per_epoch,
        decay_steps,
        alpha=0.0,
        name=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            global_step=global_step,
            name=name,
            steps_per_epoch=steps_per_epoch,
        )
        self.decay_steps = decay_steps
        self.alpha = alpha

    def build_lr_scheduler(self):
        return tf.train.cosine_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            alpha=self.alpha,
            name=self.name,
        )


class CosineDecayRestarts(BaseLRScheduler):
    def __init__(
        self,
        learning_rate,
        global_step,
        steps_per_epoch,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        name=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            global_step=global_step,
            name=name,
            steps_per_epoch=steps_per_epoch,
        )
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha

    def build_lr_scheduler(self):
        return tf.train.cosine_decay_restarts(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            first_decay_steps=self.first_decay_steps,
            t_mul=self.t_mul,
            m_mul=self.m_mul,
            alpha=self.alpha,
            name=self.name,
        )
