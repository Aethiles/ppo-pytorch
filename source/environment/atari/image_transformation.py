import cv2
import gym
import numpy as np

from source.utilities.errors import ShapeError
from typing import Tuple


class ImageTransformationEnv(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Wrapper,
                 target_size: Tuple[int, int] = (84, 84),
                 interpolation: int = cv2.INTER_AREA,
                 grayscale: bool = True,
                 ):
        """
        Resizes the observation and converts it to grayscale if demanded.
        Slightly modified from OpenAI baselines AtariWrappers. As detailed in Mnih et al. (2015) -- aka Nature paper.
        :param env: the inner environment
        :param target_size: the new image size as (width, height)
        :param interpolation: the interpolation method
        :param grayscale: bool indicating whether to convert to grayscale
        """
        super().__init__(env)
        self.target_size = target_size
        self.interpolation = interpolation
        self.grayscale = grayscale

    def observation(self,
                    observation: np.ndarray,
                    ) -> np.ndarray:
        """
        Transforms the observation by resizing and optionally converting to grayscale.
        :param observation: the observation to transform
        :return: the transformed observation
        """
        if self.grayscale:
            observation = self.convert_to_grayscale(observation)
        observation = self.resize(observation)
        # if self.grayscale:
        #     observation = np.expand_dims(observation, 0)
        return observation / 255.0

    @staticmethod
    def convert_to_grayscale(image: np.ndarray,
                             ) -> np.ndarray:
        """
        Converts the given BGR image to grayscale
        :param image: an image shaped (height, width, channels) with 3 channels
        :return: a (height, width) shaped grayscale image
        """
        if image.dtype != np.uint8:
            raise TypeError('Expected image with type np.uint8 but got {}'.format(image.dtype))
        if len(image.shape) != 3:
            raise ShapeError('Expected image shape (height, width, channels) but got {}'.format(image.shape))
        if image.shape[2] != 3:
            raise ShapeError('Expected 3 channels but got {}'.format(image.shape[2]))

        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def resize(self,
               image: np.ndarray,
               ) -> np.ndarray:
        """
        Resizes the image to the configured (height, width) shape
        :param image: the image to resize
        :return: the resized image
        """
        return cv2.resize(image, self.target_size, interpolation=self.interpolation)
