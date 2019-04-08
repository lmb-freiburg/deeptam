from abc import ABC, abstractmethod
import tensorflow as tf

class MappingNetworkBase(ABC):

    def __init__(self, batch_size=1, width=128, height=96):
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._placeholders = {
            'depth_key': tf.placeholder(tf.float32, shape=(batch_size, 1, height, width)),
            'image_key': tf.placeholder(tf.float32, shape=(batch_size, 3, height, width)),
            'image_current': tf.placeholder(tf.float32, shape=(batch_size, 3, height, width)),
            'intrinsics': tf.placeholder(tf.float32, shape=(batch_size, 4)),
            'rotation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
            'translation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
        }
        # will be filled in the derived class
        self._placeholders_state = {}


    @property
    def placeholders(self):
        """All placeholders required for feeding this network"""
        return self._placeholders

    @property
    def placeholders_state(self):
        """Placeholders which define the state of the network"""
        return self._placeholders_state

    @abstractmethod
    def build_net(self, depth_key, image_key, image_current, intrinsics, rotation, translation, state):
        """Build the mapping network

        depth_key: the current key frame depth map
        image_key: the image of the key frame
        image_current: the current image
        intrinsics: the camera intrinsics
        rotation: the current guess for the camera rotation as angle axis representation
        translation: the current guess for the camera translation
        band_scale: the scale factor for band_width, per pixel

        Returns all network outputs as a dict
        The following must be returned:

            predict_depth : Tensor
            state : dict of Tensors

        """
        pass


    @abstractmethod
    def get_state_init(self):
        """Returns a dictionary with ops for initializing the network state"""
        return {}


    @abstractmethod
    def get_state_init_np(self):
        """Returns a dictionary with numpy arrays for initializing the network state"""
        return {}

    def get_state_feed_dict(self, state):
        return {self.placeholders_state[x]: v for x, v in state.items()}



