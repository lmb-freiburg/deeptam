import tensorflow as tf
from collections import deque
from deeptam_mapper.utils.helpers import *
from deeptam_mapper.utils.datatypes import *
import numpy as np


class Mapper():
    def __init__(self,
                checkpoints,
                intrinsics,
                mapping_module_path,
                input_image_size,
                num_nb_iters=5,
                num_keep_frames=10):
        
        """
            checkpoints: a list of str
            
            intrinsics: np.array
                normalized intrinsics
                
            mapping_module_path: str
                path to the file networks.py
            
            input_image_size: (int, int)
                (height, width)
                
            num_nb_iters: int
                narrow band iterations
            
            num_keep_frames: int
                the maximum number of frames used for keyframe depth computation
        """
        
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction = 0.8
        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options))
        self._session.run(tf.global_variables_initializer())
        
        
        self._width = input_image_size[1]
        self._height = input_image_size[0]
        self._batch_size = 1
        self._intrinsics = np.squeeze(intrinsics)[np.newaxis,:]
        self._num_keep_frames = num_keep_frames
        self._num_nb_iters = num_nb_iters
        self._checkpoints = checkpoints
        
        self._mapping_module_path = mapping_module_path
        self._mapping_mod = load_myNetworks_module_noname(mapping_module_path)
        
        # cost volume for fixed band module
        self._cv_fb_generate_net = self._mapping_mod.CVGenerateNetwork(batch_size=self._batch_size,
                                                                        width=self._width,
                                                                        height=self._height,
                                                                        depth_scale_array=np.linspace(0.01,2.5,32))
        self._cv_fb_generate_placeholders= self._cv_fb_generate_net.placeholders
        self._cv_fb_generate_outputs = self._cv_fb_generate_net.build_net(**self._cv_fb_generate_net.placeholders,
                                                                          state=self._cv_fb_generate_net.placeholders_state)

    
        # cost volume for narrow band module
        self._cv_nb_generate_net = self._mapping_mod.CVGenerateNetwork(batch_size=self._batch_size,
                                                                        width=self._width,
                                                                        height=self._height,
                                                                        depth_scale_array=np.linspace(0.8,1.2,32))
        self._cv_nb_generate_placeholders = self._cv_nb_generate_net.placeholders
        self._cv_nb_generate_outputs = self._cv_nb_generate_net.build_net(**self._cv_nb_generate_net.placeholders,
                                                                          state=self._cv_nb_generate_net.placeholders_state)
        
        # fixed band module
        self._fb_depth_net = self._mapping_mod.MappingFBNetwork(batch_size=self._batch_size,
                                                               width=self._width,
                                                               height=self._height)
        self._fb_depth_placeholders = self._fb_depth_net.placeholders
        self._fb_depth_outputs = self._fb_depth_net.build_net(**self._fb_depth_net.placeholders,
                                                              state=self._fb_depth_net.placeholders_state)
    
        # narrow band module
        self._nb_depth_net = self._mapping_mod.MappingNBNetwork(batch_size=self._batch_size,
                                                               width=self._width,
                                                               height=self._height)
        self._nb_depth_placeholders = self._nb_depth_net.placeholders
        self._nb_depth_outputs = self._nb_depth_net.build_net(**self._nb_depth_net.placeholders,
                                                              state=self._nb_depth_net.placeholders_state)
        
        # narrow band refine module
        self._nb_refine_depth_net = self._mapping_mod.MappingNBRefineNetwork(batch_size=self._batch_size,
                                                                            width=self._width,
                                                                            height=self._height)
        self._nb_refine_depth_placeholders = self._nb_refine_depth_net.placeholders
        self._nb_refine_depth_outputs = self._nb_refine_depth_net.build_net(**self._nb_refine_depth_net.placeholders,
                                                                            state=self._nb_refine_depth_net.placeholders_state)
        
        
        self._keyframe = None
        self._frames = deque(maxlen=self._num_keep_frames)
   
        self._load_checkpoints()
            
    def clear(self):
        """clear the keyframe and frame list
        """
        self._keyframe = None
        self._frames = deque(maxlen=self._num_keep_frames)
        
    
    def _load_checkpoints(self):
        """loads the weights
        """
        for checkpoint in self._checkpoints:
            print('loading {0}'.format(checkpoint))
            optimistic_restore(self._session, checkpoint, verbose=True)
            
    @staticmethod
    def _compute_relative_pose(pose1, pose2):
        """Computes the relative pose for pose2 with respect to pose1
        
        pose1: Pose
            global pose which is the new reference
            
        pose1: Pose
            global pose
        """
        # compute relative transformation to key frame
        R_relative = pose2.R * pose1.R.transpose()
        t_relative = pose2.t - R_relative*pose1.t
        return Pose(R=R_relative, t=t_relative)
    
    def _create_cv_fb(self):
        """creates the fixed band costvolume using self._keyframe and self._frames.
        
        Assumes that self._frames dosen't contain the key frame 
        """
        cv = None
        conf_sum = None
        for frame_i, frame in enumerate(self._frames):
            pose = self._compute_relative_pose(self._keyframe.pose, frame.pose)
            feed_dict = {
                    self._cv_fb_generate_placeholders['depth_key']: np.ones([1, 1, self._height, self._width]),
                    self._cv_fb_generate_placeholders['image_key']: self._keyframe.image,
                    self._cv_fb_generate_placeholders['image_current']: frame.image,
                    self._cv_fb_generate_placeholders['intrinsics']: self._intrinsics,
                    self._cv_fb_generate_placeholders['rotation']: rotation_matrix_to_angleaxis(pose.R)[np.newaxis,:].astype(np.float32),
                    self._cv_fb_generate_placeholders['translation']: np.array(pose.t, dtype=np.float32)[np.newaxis,:],
            }
#             _start = time.time()
            cv_generate_out = self._session.run(self._cv_fb_generate_outputs, feed_dict=feed_dict)
#             self._network_runtimes['costvolume_fb'].append(time.time()-_start)

            if cv is None:
                cv = np.zeros_like(cv_generate_out['cv'])
                conf_sum = np.zeros_like(cv_generate_out['cv_conf'])
            cv += cv_generate_out['cv']*cv_generate_out['cv_conf']
            conf_sum += cv_generate_out['cv_conf']

        cv /= conf_sum
        cv[np.logical_not(np.isfinite(cv))] = 0

        return { 
            'cv': cv, 
            'depth_label_tensor': cv_generate_out['depth_label_tensor'],
            }

    
    def _create_cv_nb(self):
        """create the narrow band costvolume around self._keyframe.depth using self._keyframe and self._frames
        
        Assumes that self._frames dosen't contain the key frame 
        """
        cv = None
        conf_sum = None
        for frame_i, frame in enumerate(self._frames):
            pose = self._compute_relative_pose(self._keyframe.pose, frame.pose)
            feed_dict = {
                    self._cv_nb_generate_placeholders['depth_key']: self._keyframe.depth,
                    self._cv_nb_generate_placeholders['image_key']: self._keyframe.image,
                    self._cv_nb_generate_placeholders['image_current']: frame.image,
                    self._cv_nb_generate_placeholders['intrinsics']: self._intrinsics,
                    self._cv_nb_generate_placeholders['rotation']: rotation_matrix_to_angleaxis(pose.R)[np.newaxis,:].astype(np.float32),
                    self._cv_nb_generate_placeholders['translation']: np.array(pose.t, dtype=np.float32)[np.newaxis,:],
            }
#             _start = time.time()
            cv_generate_out = self._session.run(self._cv_nb_generate_outputs, feed_dict=feed_dict)
#             self._network_runtimes['costvolume_nb'].append(time.time()-_start)

            if cv is None:
                cv = np.zeros_like(cv_generate_out['cv'])
                conf_sum = np.zeros_like(cv_generate_out['cv_conf'])
            cv += cv_generate_out['cv']*cv_generate_out['cv_conf']
            conf_sum += cv_generate_out['cv_conf']

        cv /= conf_sum
        cv[np.logical_not(np.isfinite(cv))] = 0

        return { 
            'cv': cv, 
            'depth_label_tensor': cv_generate_out['depth_label_tensor'],
            }
    
    def _compute_depth_fb(self):
        """estimates the depth(fb) using self._keyframe and self._frames
        
        """
        
        cv = self._create_cv_fb()
        
        feed_dict = {
            self._fb_depth_net.placeholders['image_key']: self._keyframe.image,
            self._fb_depth_net.placeholders_state['cv']: cv['cv'],
            self._fb_depth_net.placeholders_state['depth_label_tensor']: cv['depth_label_tensor']
        }
        fb_out = self._session.run(self._fb_depth_outputs, feed_dict=feed_dict)
        
        return fb_out['predict_depth']
    
    def _compute_depth_nb(self, refine):
        """estimates the depth(nb) using self._keyframe and self._frames
        
        refine:  bool
            use the depth_nb_refine module if True
        """
        
        cv = self._create_cv_nb()
        # narrow band 
        feed_dict = {
            self._nb_depth_placeholders['image_key']: self._keyframe.image,
            self._nb_depth_net.placeholders_state['cv']: cv['cv'],
            self._nb_depth_net.placeholders_state['depth_label_tensor']: cv['depth_label_tensor'],
        }
        
        nb_out = self._session.run(self._nb_depth_outputs, feed_dict=feed_dict)
        depth_pr = nb_out['predict_depth']
        
        if refine:
            # narrow band refine
            feed_dict = {
                self._nb_refine_depth_net.placeholders['image_key']:self._keyframe.image,
                self._nb_refine_depth_net.placeholders['depth_key']: depth_pr,
                self._nb_refine_depth_net.placeholders_state['cv']:cv['cv']
            }

            nb_refine_out = self._session.run(self._nb_refine_depth_outputs, feed_dict=feed_dict)
            depth_pr = nb_refine_out['predict_depth']

        return depth_pr

    def _set_keyframe(self, image, pose):
        """sets the keyframe
        
        image: np.array
                rgb normalized in range[-0.5, 0.5], in NCHW format
        
        pose: Pose
        """
        
        self._keyframe = Frame(image=image, pose=pose)
        
    
    def _add_curframe(self, image, pose):
        """adds current frame to the _frames list
        
        image: np.array
                rgb normalized in range[-0.5, 0.5], in NCHW format
                
        pose: Pose
        """
        # naively taking all incoming frames, delete old frame when it exceeds the length limit, ideally replaced with some smart frame selection method
        if len(self._frames) == self._num_keep_frames:
            self._frames.rotate(-1)
            self._frames[-1].image = image
            self._frames[-1].pose = pose
        else:
            self._frames.append(Frame(image=image, pose=pose))
        
    
    def feed_frame(self, image, pose, keyframe_flag):
        """feed frame
        
        image: np.array
                rgb normalized in range[-0.5, 0.5], in NCHW format
                
        pose: Pose
        
        keyframe_flag: bool 
            set self._keyframe if True, else add to self._frames
        """
        if keyframe_flag:
            self._set_keyframe(image=image, pose=pose)
        else:
            self._add_curframe(image=image, pose=pose)
        
    def compute_keyframe_depth(self):
        """ estimates the depth (fb+nb(w/refine)) using self._keyframe and self._frames
        
        First run fixed band module once, and then run narrow band module and its refinement module iteratively.
        """
        
        # fixed band
        self._keyframe.depth = self._compute_depth_fb()
        
        # narrow band
        for i in range(self._num_nb_iters):
            # do not run the refine net on the last iteration
            refine = i < self._num_nb_iters
            self._keyframe.depth = self._compute_depth_nb(refine)
        return self._keyframe.depth
            
