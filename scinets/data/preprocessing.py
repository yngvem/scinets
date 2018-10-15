"""

"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np


class Preprocessor:
    def __call__(self, idx, images, target):
        return idx, images, target


class PreprocessorJoiner(Preprocessor):
    def __init__(self, preprocessor_dicts):
        self.preprocessors = [get_operator(preprocessor_dict)
                                for preprocessor_dict in preprocessor_dicts]
    
    def __call__(self, idx, images, target):
        images = np.concatenate(
            [preprocessor(images) for preprocessor in self.preprocessors],
            axis=-1
        )


class Windowing(Preprocessor):
    def __init__(self, window_center, window_width):
        self.window_center, self.window_width = window_center, window_width

    def __call__(self, idx, images, target):
        images = images - window_width
        c_min = self.window_center - self.window_width/2
        c_max = self.window_center + self.window_width/2
        images[images < c_min] = c_min
        images[images > c_max] = c_max
        return idx, images, target


class HoundsfieldWindowing(Preprocessor):
    def __init__(self, window_center, window_width, ct_channel=0,
                 houndsfield_offset=1024):
        self.ct_channel = ct_channel
        self.houndsfield_offset = houndsfield_offset
        super().__init__(window_center, window_width)

    def __call__(self, idx, images, target):
        images = images.copy()
        images[self.ct_channel] -= self.houndsfield_offset
        _, images[self.ct_channel], _ = super().__call__(
            idx,
            images[self.ct_channel],
            target
        )

        return idx, images, target


class MultipleHoundsfieldWindowing(HoundsfieldWindowing):
    def __init__(self, window_centers, window_windths, ct_channel=0,
                 houndsfield_offset=1024):
        self.window_centers = window_centers
        self.window_widths = window_widths
        self.ct_channel = ct_channel
        self.houndsfield_offset = houndsfield_offset

    def generate_all_houndsfield_windowing_channels(images):
        images = images.copy()
        new_ct_channels = []
        for window_center, window_width in zip(self.window_centers, 
                                               self.window_widths):
            self.window_center, self.window_width = window_center, window_width
            _, new_channel, _ = super().__call__(idx, images, target)
            new_ct_channels.append(new_channel)

        return np.stack(new_ct_channels, axis=-1)

    def __call__(self, idx, images, target):
        new_ct_channels = generate_all_houndsfield_windowing_channels(images)

        # Replace current CT channel with all windowed versions
        images = np.delete(images, ct_channel, axis=-1)
        images = np.concatenate((images, new_ct_channels), axis=-1)
        return idx, images, target







if __name__ == '__main__':
    pass

