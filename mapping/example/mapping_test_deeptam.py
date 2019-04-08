import tensorflow as tf
from deeptam_mapper.models.helpers import *
from deeptam_mapper.utils.helpers import *
from deeptam_mapper.utils.datatypes import *
import deeptam_mapper.evaluation.metrics as metrics
from deeptam_mapper.utils.vis_utils import convert_array_to_colorimg,convert_array_to_grayimg
import sys
import numpy as np
import time
import os 
from collections import namedtuple
DepthMetrics = namedtuple('DepthMetrics', ['l1_inverse', 'scale_invariant', 'abs_relative'])
import pickle
import matplotlib.pyplot as plt

def init_visualization(title):
    """Initializes a simple visualization for tracking
    
    title: str
    """
    fig = plt.figure()
    fig.set_size_inches(10.5, 8.5)
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(2,2,1)
    
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Keyframe')
           
    ax2 = fig.add_subplot(2,2,2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax2.set_title('Current frame: ')
    ax3 = fig.add_subplot(2,2,3)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    ax3.set_title('FB depth pr: frame ')
    ax4 = fig.add_subplot(2,2,4)
    
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title('NB depth pr: iter ')
    
    return [ax1, ax2, ax3, ax4]


def update_visualization(axes, 
                         image_cur,
                         image_cur_virtual,
                         depth_pr_fb,
                         depth_pr_nb,
                         title_suffixes):
    """ Updates the visualization for tracking
    
    axes: a list of plt.axes
    
    image_cur, image_cur_virtual: np.array
    
    depth_pr_fb, depth_pr_nb: np.array
    
    title_subfixes: a list of str
    
    """

    if image_cur_virtual is not None:
        image_cur = convert_array_to_colorimg(image_cur.squeeze())
        image_cur_virtual = convert_array_to_colorimg(image_cur_virtual.squeeze()) 
        depth_pr_vis_fb = convert_array_to_grayimg(depth_pr_fb.squeeze()) 
        if depth_pr_nb is not None:
            depth_pr_vis_nb = convert_array_to_grayimg(depth_pr_nb.squeeze()) 

        axes[0].imshow(np.array(image_cur))       
        axes[1].imshow(np.array(image_cur_virtual))
        axes[2].imshow(np.array(depth_pr_vis_fb))
        if depth_pr_nb is not None:
            axes[3].imshow(np.array(depth_pr_vis_nb))
            
        axes[0].set_title('Keyframe')
        axes[1].set_title('Current frame: ' + title_suffixes[1])
        axes[2].set_title('FB depth pr: frame '+ title_suffixes[2])
        axes[3].set_title('NB depth pr: iter ' + title_suffixes[3])

    plt.pause(0.5)

def compute_depth_metrics(pr, gt):
    """Computes depth errors
    pr: np.ndarray 
        The prediction as absolute depth values

    gt: np.ndarray
        The ground truth as absolute depth values
    """
    valid_mask = metrics.compute_valid_depth_mask(pr, gt)
    valid_pr = pr[valid_mask]
    valid_gt = gt[valid_mask]

    return DepthMetrics(**
        {
            'l1_inverse': metrics.l1_inverse(valid_pr, valid_gt),
            'scale_invariant': metrics.scale_invariant(valid_pr, valid_gt),
            'abs_relative': metrics.abs_relative(valid_pr, valid_gt),
        })


def create_cv_conf_from_sequence_py(depth_key,
                                    sub_seq_py,
                                    session,
                                    net,
                                    outputs):
    """Compute
    
    depth_key: np.array
    
    sub_seq_py: SeqPy
    
    session: tf.Session
    
    net:  mapping_mod.CVGenerateNetwork
    
    outputs: dict of Tensor
    """
    
    frame_end = sub_seq_py.seq_len - 1
    cv_list = []
    cv_conf_list = []
    
    # computes pairwise cost volume with its conf between the keyframe and all the other frames in the sequence
    for frame in range(1, frame_end + 1):
        feed_dict = {
                net.placeholders['depth_key']: depth_key,
                net.placeholders['image_key']: sub_seq_py.get_image(frame=0),
                net.placeholders['image_current']:sub_seq_py.get_image(frame=frame),
                net.placeholders['intrinsics']:np.expand_dims(sub_seq_py.intrinsics,axis=0),
                net.placeholders['rotation']:np.expand_dims(sub_seq_py.get_rotation(frame=frame),axis=0),
                net.placeholders['translation']:np.expand_dims(sub_seq_py.get_translation(frame=frame),axis=0),
        }
        cv_generate_out = session.run(outputs, feed_dict=feed_dict)
        cv = cv_generate_out['cv']
        cv_conf = cv_generate_out['cv_conf']
        depth_label_tensor = cv_generate_out['depth_label_tensor']
        cv_list.append(cv)
        cv_conf_list.append(cv_conf)
    
    # cost volume aggregation over all the frames in the sequence
    cv_sum = np.zeros_like(cv)
    cv_conf_sum = np.zeros_like(cv_conf)
    for ind in range(len(cv_list)):
        cv_sum += cv_list[ind]*cv_conf_list[ind]
        cv_conf_sum += cv_conf_list[ind]
    cv_mean = cv_sum/cv_conf_sum
    cv_mean[cv_mean == np.inf] = 0
    cv_mean[cv_mean == -np.inf] = 0
    
    return np.nan_to_num(cv_mean), depth_label_tensor

def mapping_with_pose(
                        datafile,
                        mapping_mod_path,
                        checkpoints,
                        gpu_memory_fraction=None,
                        width = 320,
                        height = 240,
                        max_sequence_length=10,
                        nb_iterations_num=5,
                        savedir=None):
    
    tf.reset_default_graph()
    
    gpu_options = tf.GPUOptions()
    if not gpu_memory_fraction is None:
        gpu_options.per_process_gpu_memory_fraction=gpu_memory_fraction
    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    
    # build mapping mod
    mapping_mod = load_myNetworks_module_noname(mapping_mod_path)

    # fixed band module
    fb_depth_net = mapping_mod.MappingFBNetwork(batch_size=1, width=width, height=height)
    fb_depth_outputs = fb_depth_net.build_net(**fb_depth_net.placeholders,state=fb_depth_net.placeholders_state)

    # narrow band module
    nb_depth_net = mapping_mod.MappingNBNetwork(batch_size=1, width=width, height=height)
    nb_depth_outputs = nb_depth_net.build_net(**nb_depth_net.placeholders,state=nb_depth_net.placeholders_state)

    # narrow band refinement module
    nb_refine_depth_net = mapping_mod.MappingNBRefineNetwork(batch_size=1, width=width, height=height)
    nb_refine_depth_outputs = nb_refine_depth_net.build_net(**nb_refine_depth_net.placeholders,state=nb_refine_depth_net.placeholders_state)

    # pairwise cost volume generation for fixed band module
    cv_fb_generate_net = mapping_mod.CVGenerateNetwork(batch_size=1, width=width, height=height, depth_scale_array=np.linspace(0.01,2.5,32))
    cv_fb_generate_outputs = cv_fb_generate_net.build_net(**cv_fb_generate_net.placeholders,state=cv_fb_generate_net.placeholders_state)

    # pairwise cost volume generation for narrow band module
    cv_nb_generate_net = mapping_mod.CVGenerateNetwork(batch_size=1, width=width, height=height, depth_scale_array=np.linspace(0.8,1.2,32))
    cv_nb_generate_outputs = cv_nb_generate_net.build_net(**cv_nb_generate_net.placeholders,state=cv_nb_generate_net.placeholders_state)
    
    
    # load weights
    session.run(tf.global_variables_initializer())

    for checkpoint in checkpoints:
        optimistic_restore(session,checkpoint,verbose=True)
        
    # read input data
    with open(datafile,'rb') as f:
        sub_seq_py = pickle.load(f)

    axes = init_visualization('DeepTAM_Mapper')
    ######### depth_gt
    depth_gt = sub_seq_py.get_depth(frame=0)


    ######### fixed band prediction with increasing number of frames
    depth_init = np.ones([1,1,240,320])

    for frame_id in range(1,sub_seq_py.seq_len):

        frame = frame_id
        sub_sub_seq_py = SubSeqPy(sub_seq_py, start_frame=0, seq_len=frame+1)
 

        cv, depth_label_tensor = create_cv_conf_from_sequence_py(depth_init, sub_sub_seq_py, session, cv_fb_generate_net, cv_fb_generate_outputs)
        feed_dict = {
            fb_depth_net.placeholders['image_key']: sub_seq_py.get_image(frame=0),
            fb_depth_net.placeholders_state['cv']:cv,
            fb_depth_net.placeholders_state['depth_label_tensor']:depth_label_tensor,
        }
        fb_out = session.run(fb_depth_outputs, feed_dict=feed_dict)

        depth_pr = fb_out['predict_depth']
            
        dm = compute_depth_metrics(1/depth_gt, 1/depth_pr)
        update_visualization(axes, 
                             sub_seq_py.get_image(0),
                             sub_seq_py.get_image(frame=frame),
                             fb_out['predict_depth'],
                             None,
                             ['',str(frame_id),str(frame_id),''])
        
    ######### narrow band prediction with increasing number of iterations 
    for iteration in range(nb_iterations_num):

        cv, depth_label_tensor = create_cv_conf_from_sequence_py(depth_pr, sub_seq_py, session, cv_nb_generate_net, cv_nb_generate_outputs)
        feed_dict = {
            nb_depth_net.placeholders['image_key']: sub_seq_py.get_image(frame=0),
            nb_depth_net.placeholders_state['cv']:cv,
            nb_depth_net.placeholders_state['depth_label_tensor']:depth_label_tensor,
        }
        nb_out = session.run(nb_depth_outputs, feed_dict=feed_dict)
        depth_pr = nb_out['predict_depth']

        feed_dict = {
            nb_refine_depth_net.placeholders['image_key']: sub_seq_py.get_image(frame=0),
            nb_refine_depth_net.placeholders['depth_key']: depth_pr,
            nb_refine_depth_net.placeholders_state['cv']:cv,
        }
        nb_refine_out = session.run(nb_refine_depth_outputs, feed_dict=feed_dict)
        depth_pr = nb_refine_out['predict_depth'] 
        
        dm = compute_depth_metrics(1/depth_gt, 1/depth_pr)
        update_visualization(axes,
                             sub_seq_py.get_image(0),
                             sub_seq_py.get_image(frame=frame),
                             fb_out['predict_depth'], 
                             nb_refine_out['predict_depth'],
                             ['',str(frame_id),str(frame_id),str(iteration)])    
            
    plt.show()
    del session
    tf.reset_default_graph()



def main():

    
    examples_dir = os.path.dirname(__file__)
    mapping_module_path = os.path.join(examples_dir,'..','python/deeptam_mapper/models/networks.py')
    checkpoints = [os.path.join(examples_dir, '..', 'weights', 'deeptam_mapper_weights','snapshot-800000'),
                    ]

    sequence_len = 10
    nb_iterations_num = 5

    width = 320
    height = 240

    datafile = os.path.join(examples_dir,'..','data/sun3d_example_seq.pkl')
    mapping_with_pose(
                    datafile,
                    mapping_module_path,
                    checkpoints,
                    max_sequence_length=sequence_len,
                    nb_iterations_num=nb_iterations_num,
                    width=width,
                    height=height,
                    )
    
    
if __name__ == '__main__':

    main()
