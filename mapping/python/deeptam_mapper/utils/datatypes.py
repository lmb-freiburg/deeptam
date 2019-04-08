from collections import namedtuple

# depth always stores the absolute depth values (not inverse depth)
# image is a PIL.Image with the same dimensions as depth
# depth_metric should always be 'camera_z'
# K corresponds to the width and height of image/depth
# R, t is the world to camera transform
View = namedtuple('View',['R','t','K','image','depth','depth_metric'])


# stores a camera pose
# R, t is the world to camera transform
Pose = namedtuple('Pose',['R','t'])
from minieigen import Matrix3, Vector3
from .rotation_conversion import *
import numpy as np

def Pose_identity():
    """Returns the identity pose"""
    return Pose(R = Matrix3.Identity, t = Vector3.Zero)

class SeqPy:
    """This class is used for handling sequence data in format of python
    
    """
    def __init__(self,seq_len):
        """Initialize an empty sequence
        
            seq_len: int
                sequence length
        """
        self.depth_list = [None] * seq_len
        self.image_list = [None] * seq_len
        self.flow_list = [None] * seq_len
        self.pose_list = [None] * seq_len
        self.intrinsics = None
        
        self.seq_len = seq_len
            
    def set_pose_list(self, translation_list, rotation_list):
        """Set pose list
        
            translation_list: a list of numpy array 
            
            rotation_list: a list of numpy array        
        """
        assert(len(translation_list) == self.seq_len), 'cannot initialize pose from translation list. sequence length is not compatible'
        assert(len(rotation_list) == self.seq_len), 'cannot initialize pose from rotation list. sequence length is not compatible'
        
        for frame in range(self.seq_len):
            rot = rotation_list[frame].squeeze()
            trans = translation_list[frame].squeeze()
            R = Matrix3(angleaxis_to_rotation_matrix(numpy_to_Vector3(rot)))
            t = numpy_to_Vector3(trans)
            pose = Pose(R=R,t=t)
            self.pose_list[frame] = pose
    
    def set_flow_list(self, flow_list):
        """Set flow list
        
            flow_list: a list of numpy array
        """
        assert(len(flow_list) == self.seq_len), 'cannot initialize pose from flow list. sequence length is not compatible'
        for frame in range(self.seq_len):
            self.flow_list[frame] = flow_list[frame]
      
    def set_flow_from_depth(self):
        """Set flow list using depth and camera pose
        
        """
        for frame in range(self.seq_len):
            self.flow_list[frame] = self.compute_flow_from_depth(0,frame)
    
            
    def set_image_list(self, image_list):
        """Set image list
        
            image_list: a list of numpy array
        """
        assert(len(image_list) == self.seq_len), 'cannot initialize pose from image list. sequence length is not compatible'
        for frame in range(self.seq_len):
            self.image_list[frame] = image_list[frame]
            
    def set_intrinsics(self, intrinsics):
        """Set intrinsics
        
            intrinsics: numpy array
        """
        self.intrinsics = intrinsics
            
    def set_depth_list(self, depth_list):
        """Set depth list
        
            depth_list: a list of numpy array
        """
        for frame in range(len(depth_list)):
            self.depth_list[frame] = depth_list[frame]
        
    def get_rotation(self, frame, rotation_format='ANGLEAXIS'):
        """Get rotation at certain frame 
        
            frame: int
            
            rotation_format: str 
                'ANGLEAXIS','ROTATION_MATRIX'
                
        """
        if rotation_format == 'ROTATION_MATRIX':
            return np.array(self.pose_list[frame].R)
        elif rotation_format == 'ANGLEAXIS':
            angleaxis = rotation_matrix_to_angleaxis(self.pose_list[frame].R)
            return angleaxis
        
    def get_translation(self, frame, normalize=False):
        """Get translation at certain frame
        
            frame: int
            
            normalize: bool
        """
        if normalize:
            trans = self.pose_list[frame].t / self.pose_list[frame].t.norm()
        else:
            trans = self.pose_list[frame].t
        return np.array(trans)
    
    def get_flow(self, frame):
        """Get flow at certain frame
        
            frame: int
        """
        return self.flow_list[frame]
    
    def get_image(self, frame):
        """Get image at certain frame
        
            frame: int
        """
        return  self.image_list[frame]
    
    def get_K(self, normalize=False):
        """Get camera intrinsics matrix K 
        
            normalize: bool 
        
        """
        if normalize:
            width = 1
            height = 1
        else:
            if self.get_depth(0) is not None:
                width = self.get_depth(0).shape[-1]
                height = self.get_depth(0).shape[-2]
                
        fx = self.intrinsics[0]*width
        fy = self.intrinsics[1]*height
        cx = self.intrinsics[2]*width
        cy = self.intrinsics[3]*height
        
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)
    
    def get_depth(self, frame, inverse=False):
        """Get depth at certain frame
        
            frame: int
            
            inverse: bool 
                inverse the depth
        """
        if inverse:
            return 1/self.depth_list[frame]
        else:
            return self.depth_list[frame]

    
    def adjust_pose_wrt_ref(self, ref):
        """Adjust the poses with respect to the reference frame. 
        
            ref: int
        """
        assert(self.pose_list[0]), 'pose list does not exist'
        
        adjusted_poses = [None]*len(self.pose_list)
    
        pose_ref = self.pose_list[ref]

        R_ref = pose_ref.R
        t_ref = pose_ref.t

        for ind, pose in enumerate(self.pose_list):
            if ind == ref:
                adjusted_poses[ind] = Pose(R = Matrix3.Identity, t = Vector3.Zero)
            else:
                R_adjusted = pose.R * R_ref.transpose()
                t_adjusted = pose.t - R_adjusted*t_ref
                adjusted_poses[ind] = Pose(R = R_adjusted, t= t_adjusted)
                
        self.pose_list = adjusted_poses
        
    def offset_pose(self, offset):
        """Offset the pose list with a certain pose. 
        
            inc_pose: Pose
        """
        offseted_poses = [None]*len(self.pose_list)
        
        for frame in range(self.seq_len):
            new_R = self.pose_list[frame].R * offset.R
            new_t = self.pose_list[frame].R * offset.t+self.pose_list[frame].t
            offseted_poses[frame] = Pose(R=new_R,t=new_t)
            
#         self.pose_list = offseted_poses
        return offseted_poses
    
                
class SubSeqPy(SeqPy):
    def __init__(self, sequence, start_frame=0, seq_len=2):
        
        
        self.seq_len = seq_len
        
        self.image_list = sequence.image_list[start_frame:start_frame+seq_len]
        self.depth_list = sequence.depth_list[start_frame:start_frame+seq_len]
        self.flow_list = [None] * seq_len
        self.pose_list = sequence.pose_list[start_frame:start_frame+seq_len]
        self.intrinsics = sequence.intrinsics
        
def generate_sub_seq_list_py(sequence, sub_seq_len=2):
    """Generate sub sequence list 
    
        sequence: SeqPy class
        
        sub_seq_len: int
    
        Returns a list of SubSeqPy class object(aligned with the first frame of the subseq) and a list of View tuple(aligned with the first frame of sequence)
    """
    sub_seq_list = []
    total_seq_len = sequence.seq_len
    seq_start = 0
    key_view_list = []
    while(seq_start <= total_seq_len - sub_seq_len):
        sub_seq = SubSeqPy(sequence, seq_start, sub_seq_len)
        sub_seq.adjust_pose_wrt_ref(0)
        
        
        view = View(R=sequence.get_rotation(frame=seq_start,rotation_format='ROTATION_MATRIX'),
            t=sequence.get_translation(frame=seq_start),
            K=sequence.get_K(normalize=False),
            image=convert_image(sequence.get_image(frame=seq_start).squeeze()),
            depth=sequence.get_depth(frame=seq_start,inverse=True).squeeze(),
            depth_metric='camera_z')
        key_view_list.append(view)
        
        seq_start = seq_start + sub_seq_len - 1
        sub_seq_list.append(sub_seq)


    return sub_seq_list, key_view_list
