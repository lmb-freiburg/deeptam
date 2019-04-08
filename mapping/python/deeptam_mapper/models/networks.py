from .networks_base import *
from .blocks import *
from .helpers import *
import numpy as np

class CVGenerateNetwork(MappingNetworkBase):
    def get_state_init(self):
        return {}
    
    def get_state_init_np(self):
        """Returns a dictionary with numpy arrays for initializing the network state"""
        return {}
    
    def __init__(self, batch_size=1, width=256, height=192, depth_scale_array=np.linspace(0.01,2.5,32)):
        super().__init__(batch_size=batch_size, width=width, height=height)
        state = {
                    'depth_scale_array': depth_scale_array,
            }
        self._placeholders_state.update(state)
        
    def build_net(self, depth_key, image_key, image_current, intrinsics, rotation, translation, state):
        
        depth_label_list = get_depth_label_list(depth_key, depth_scale_array=self._placeholders_state['depth_scale_array'])
        
        patch_size = 3
        border_radius = patch_size//2 + 1
        depths = tf.concat(depth_label_list, axis=1)
        warped, mask, depths = create_depthsweep_images_tensor(
                image=image_current,
                rotation=rotation,
                translation=translation, 
                intrinsics=intrinsics, 
                depth_values=depths, 
                border_radius=border_radius, 
                )
        cv, conf = compute_sad_volume_with_confidence(image_key, warped, mask)
        
        return {
                'cv':cv,
                'cv_conf': conf,
                'depth_label_tensor': depths,
                }
    
class MappingFBNetwork(MappingNetworkBase):
    def __init__(self, batch_size=1, width=256, height=192):
        super().__init__(batch_size=batch_size, width=width, height=height)
        state = {
                    'cv': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
                    'depth_label_tensor': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width))
            }
        self._placeholders_state.update(state)
    
    def get_state_init(self):
        return {
                     'cv': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
                     'depth_label_tensor': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
                }
    def get_state_init_np(self):
        """Returns a dictionary with numpy arrays for initializing the network state"""
        return {}
    def build_net(self, depth_key, image_key, image_current, intrinsics, rotation, translation, state):

        with tf.variable_scope("net_D_fb", reuse=None):
            cv_raw = state['cv']
            depth_label_list = tf.split(state['depth_label_tensor'],num_or_size_splits=32,axis=1)
            predictions = depth_fb_block(tf.concat([image_key,cv_raw],axis=1))

            depth_prediction_factor = tf.abs(tf.tanh(predictions['predict_depth0']))
            depth_prediction_value = (1-depth_prediction_factor)*depth_label_list[0] + depth_prediction_factor*depth_label_list[-1]
            depth_prediction = {
                'predict_depth0': depth_prediction_value,
                } 
            
        return {
                    'predict_depth': depth_prediction_value,
                }

class MappingNBNetwork(MappingNetworkBase):
    def __init__(self, batch_size=1, width=256, height=192):
        super().__init__(batch_size=batch_size, width=width, height=height)
        state = {
                    'cv': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
                    'depth_label_tensor': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width))
            }
        self._placeholders_state.update(state)
    
    def get_state_init(self):
        return {
                     'cv': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
                     'depth_label_tensor': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width))
                }
    def get_state_init_np(self):
        """Returns a dictionary with numpy arrays for initializing the network state"""
        return {}
    def build_net(self, depth_key, image_key, image_current, intrinsics, rotation, translation, state):

        with tf.variable_scope("net_D_iter_D_refine", reuse=None):
            depth_label_tensor = state['depth_label_tensor']
            cv_raw = state['cv']
            cv_refine = depth_nb_block(tf.concat([image_key,cv_raw],axis=1),cv_raw)
            cost_softmax = tf.nn.softmax(
                                -cv_refine,
                                dim=1,
                                name='softmax'
                            )
            depth_value = tf.reduce_sum(cost_softmax*depth_label_tensor,axis=1)
            depth_value_pr = tf.expand_dims(depth_value,axis=1)
            depth_prediction = {
                        'predict_depth0':depth_value_pr,
                        } 
                    
        return {
                    'predict_depth': depth_value_pr,
                }    
class MappingNBRefineNetwork(MappingNetworkBase):
    def __init__(self, batch_size=1, width=256, height=192):
        super().__init__(batch_size=batch_size, width=width, height=height)
        state = {
                    'cv': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
            }
        self._placeholders_state.update(state)
    
    def get_state_init(self):
        return {
                     'cv': tf.placeholder(tf.float32, shape=(batch_size, 32, height, width)),
                }
    def get_state_init_np(self):
        """Returns a dictionary with numpy arrays for initializing the network state"""
        return {}
    def build_net(self, depth_key, image_key, image_current, intrinsics, rotation, translation, state):

        with tf.variable_scope("net_D_iter_D_refine", reuse=None):
            with tf.variable_scope("net_D_single_refine", reuse=None): 
                cv_raw = state['cv']
                depth_inputs = tf.stop_gradient(tf.concat([image_key, depth_key,cv_raw],axis=1))
                depth_prediction_refine = depth_nb_refine_block(depth_inputs)

                    
        return {
                    'predict_depth': depth_prediction_refine['predict_depth0'],
                }  

