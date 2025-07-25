from penzai import pz
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class ModulaLayer(pz.nn.Layer, ABC):
    mass: float = 1.0
    sensitivity: float = 1.0

    @classmethod
    @abstractmethod
    def from_module(cls, module):
        return module

    @abstractmethod
    def normalize(self):
        return self


def modularize(module: pz.nn.Layer):
    if isinstance(module, ModulaLayer):
        return module
    return ModulaLayer.from_module(module)
