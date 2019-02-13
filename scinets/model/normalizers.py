__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


from abc import ABC, abstractmethod
from .._backend_utils import SubclassRegister
import tensorflow as tf


normalizer_register = SubclassRegister("normalizer")


def get_normalizer(normalizer):
    return normalizer_register.get_item(normalizer)


@normalizer_register.link_base
class BaseNormalizer(ABC):
    def __call__(self, x, **kwargs):
        return self._build_normalizer(x, **kwargs)

    @abstractmethod
    def _build_normalizer(self, x):
        pass


class BatchNormalization(BaseNormalizer):
    def __init__(self, training):
        self.training = training

    def _build_normalizer(self, x, name="BN"):
        return tf.layers.batch_normalization(x, training=self.training, name=name)
