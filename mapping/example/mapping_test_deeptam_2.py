#
# This script uses the Mapper class to compute the keyframe depth.
#
import tensorflow as tf
import numpy as np
from deeptam_mapper.utils.helpers import *
from deeptam_mapper.utils.datatypes import *
from deeptam_mapper.mapper import Mapper
import pickle
import matplotlib.pyplot as plt
from deeptam_mapper.utils.vis_utils import convert_array_to_colorimg,convert_array_to_grayimg



def visualization(image, depth):
    """Initializes a simple visualization for tracking
    
    image, depth: np.array
    """
    
    image = convert_array_to_colorimg(image.squeeze())
    depth = convert_array_to_grayimg(depth.squeeze()) 
    
    fig = plt.figure()
    fig.set_size_inches(10.5, 4.5)
    fig.suptitle('DeepTAM_Mapper', fontsize=16)

    ax1 = fig.add_subplot(1,2,1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('Keyframe Image')
    ax1.imshow(np.array(image)) 
           
    ax2 = fig.add_subplot(1,2,2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Depth Prediction')
    ax2.imshow(np.array(depth))
    
    plt.show()


def main():

    # set the paths
    examples_dir = os.path.dirname(__file__)
    mapping_module_path = os.path.join(examples_dir,'..','python/deeptam_mapper/models/networks.py')
    checkpoints = [os.path.join(examples_dir, '..', 'weights', 'deeptam_mapper_weights','snapshot-800000'),
                    ]
    datafile = os.path.join(examples_dir,'..','data/sun3d_example_seq.pkl')

    # load the example sequence
    with open(datafile,'rb') as f:
        seq = pickle.load(f)
    
    # intialize mapper
    num_nb_iters = 5
    num_keep_frames = 10    
    input_image_size = (240,320)
    sun3d_intrinsics = seq.intrinsics

    mapper = Mapper(checkpoints,
                sun3d_intrinsics,
                mapping_module_path,
                input_image_size,
                num_nb_iters=num_nb_iters,
                num_keep_frames=num_keep_frames)
    
    # use the first frame as keyframe and estimate its depth
    mapper.clear()
    for i in range(seq.seq_len):
        if i == 0:
            keyframe_flag = True
        else:
            keyframe_flag = False
        
        mapper.feed_frame(seq.image_list[i], seq.pose_list[i], keyframe_flag=keyframe_flag)
        
    depth_pr = mapper.compute_keyframe_depth()
    
    visualization(mapper._keyframe.image, depth_pr)
    
    del mapper
if __name__ == '__main__':

    main()
