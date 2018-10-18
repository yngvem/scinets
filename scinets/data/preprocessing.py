"""

"""


__author__ = 'Yngve Mardal Moe'
__email__ = 'yngve.m.moe@gmail.com'


import numpy as np


class Preprocessor:
    """Superclass for all preprocessors. Does nothing.
    """
    def __call__(self, images):
        """The function being applied to the input images.
        """
        return images

    def output_channels(self, input_channels):
        """The number of output channels as a function of input channels.
        """
        return input_channels


class PreprocessingPipeline(Preprocessor):
    """Create a preprocessing pipeline form a list of preprocessors.

    The output of the first preprocessor is used as argument for the second,
    and so forth. The output of the last preprocessor is then returned.
    """
    def __init__(self, preprocessor_dicts):
        def get_operator(preprocessor_dict):
            preprocessor = globals()[preprocessor_dict['operator']]
            return preprocessor(*preprocessor_dict['arguments'])

        self.preprocessors = [get_operator(preprocessor_dict)
                                for preprocessor_dict in preprocessor_dicts]
    
    def __call__(self, images):
        for preprocessor in self.preprocessors:
            images = preprocessor(images)
        return images

    def output_channels(self, input_channels):
        output_channels = input_channels
        for preprocessor in self.preprocessors:
            output_channels = preprocessor.output_channels(output_channels)
        return output_channels


class ChannelRemoverPreprocessor(Preprocessor):
    def __init__(self, channel):
        self.unwanted_channel = channel

    def __call__(self, images):
        return np.delete(images, self.unwanted_channel, axis=-1)

    def output_channels(self, input_channels):
        return input_channels-1


class WindowingPreprocessor(Preprocessor):
    """Used to set the dynamic range of an image.
    """
    def __init__(self, window_center, window_width, channel):
        self.window_center, self.window_width = window_center, window_width
        self.channel = channel

    def perform_windowing(self, image):
        image = image - self.window_center
        image[image < -self.window_width/2] = -self.window_width/2
        image[image > self.window_width/2] = self.window_width/2
        return image

    def __call__(self, images):
        images = images.copy()
        images[..., self.channel] = self.perform_windowing(
            images[..., self.channel]
        )
        return images


class MultipleWindowsPreprocessor(WindowingPreprocessor):
    """Used to create multiple windows of the same channel.
    """
    def __init__(self, window_centers, window_widths, channel):
        self.window_centers = window_centers
        self.window_widths = window_widths
        self.channel = channel

    def generate_all_windows(self, images):
        channel = images[..., self.channel]
        new_channels = []
        for window_center, window_width in zip(self.window_centers, 
                                               self.window_widths):
            self.window_center, self.window_width = window_center, window_width
            new_channel = self.perform_windowing(channel)
            new_channels.append(new_channel)

        return np.stack(new_channels, axis=-1)

    def __call__(self, images):
        new_channels = self.generate_all_windows(images)

        # Replace current CT channel with all windowed versions
        images = np.delete(images, self.channel, axis=-1)
        images = np.concatenate((images, new_channels), axis=-1)
        return images

    def output_channels(self, input_channels):
        return input_channels + len(self.window_widths) - 1



class HoundsfieldWindowingPreprocessor(WindowingPreprocessor):
    """A windowing operator, with the option to set the Houndsfield unit offset.

    The Houndsfield unit offset is simply added to the window center,
    but this makes the window centers on the same scale as what radiologists
    use.
    """
    def __init__(self, window_center, window_width, channel=0,
                 houndsfield_offset=1024):
        window_center += houndsfield_offset
        super().__init__(window_center, window_width, channel)


class MultipleHoundsfieldWindowsPreprocessor(MultipleWindowsPreprocessor):
    """Perform several windows of the CT channel with a Houndsfield unit offset.
    
    The Houndsfield unit offset is simply added to the window center,
    but this makes the window centers on the same scale as what radiologists
    use.
    """
    def __init__(self, window_centers, window_widths, channel=0,
                 houndsfield_offset=1024):
        window_centers = [wc + houndsfield_offset for wc in window_centers]
        super().__init__(window_centers, window_widths, channel)


if __name__ == '__main__':
    pass

