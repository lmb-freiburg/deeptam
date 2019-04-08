import tensorflow as tf
import lmbspecialops as sops
import numpy as np

def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.1)
    #return tf.nn.leaky_relu(x, alpha=0.1)


def default_weights_initializer():
    return tf.variance_scaling_initializer(scale=2)


def conv2d(inputs, num_outputs, kernel_size, data_format, padding=None, **kwargs):
    """Convolution with 'same' padding"""

    if padding is None:
        padding='same'
    return tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs,
        kernel_size=kernel_size,
        kernel_initializer=default_weights_initializer(),
        padding=padding,
        data_format=data_format,
        **kwargs,
        )


def convrelu(inputs, num_outputs, kernel_size, data_format, activation=None, **kwargs):
    """Shortcut for a single convolution+relu 
    
    See tf.layers.conv2d for a description of remaining parameters
    """
    if activation is None:
        activation=myLeakyRelu
    return conv2d(inputs, num_outputs, kernel_size, data_format, activation=activation, **kwargs)


def convrelu2(inputs, num_outputs, kernel_size, name, stride, data_format, padding=None, activation=None, **kwargs):
    """Shortcut for two convolution+relu with 1D filter kernels 
    
    num_outputs: int or (int,int)
        If num_outputs is a tuple then the first element is the number of
        outputs for the 1d filter in y direction and the second element is
        the final number of outputs.
    """
    if isinstance(num_outputs,(tuple,list)):
        num_outputs_y = num_outputs[0]
        num_outputs_x = num_outputs[1]
    else:
        num_outputs_y = num_outputs
        num_outputs_x = num_outputs

    if isinstance(kernel_size,(tuple,list)):
        kernel_size_y = kernel_size[0]
        kernel_size_x = kernel_size[1]
    else:
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size

    if padding is None:
        padding='same'

    if activation is None:
        activation=myLeakyRelu

    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs_y,
        kernel_size=[kernel_size_y,1],
        strides=[stride,1],
        padding=padding,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'y',
        **kwargs,
    )
    return tf.layers.conv2d(
        inputs=tmp_y,
        filters=num_outputs_x,
        kernel_size=[1,kernel_size_x],
        strides=[1,stride],
        padding=padding,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'x',
        **kwargs,
    )


def fcrelu(inputs, name, num_outputs, weights_regularizer=None, activation=None, **kwargs):
    """Shortcut for fully_connected layer + relu 
    
    See tf.layers.dense for a description of remaining parameters

    num_outputs: int 

    """
    if activation is None:
        activation=myLeakyRelu

    return tf.layers.dense(
            inputs=inputs,
            units=num_outputs,
            activation=activation,
            kernel_initializer=default_weights_initializer(),
            name=name,
            **kwargs,
            )

def compute_confidence_for_costvolume_2(costvolume,scale=10, epsilon=1e-6):    
    """Compute the confidence for a costvolume

        costvolume: Tensor 
            in NCHW format
    """
    shape = costvolume.get_shape().as_list()
    num = shape[1]
    cv_min = tf.reduce_min(costvolume,axis=1,keep_dims=True)
    tmp = (costvolume - cv_min)**2 
    conf = 1 - (tf.reduce_sum(tf.exp(-tmp*scale),axis=1,keep_dims=True)-1)/(num-1)
    return conf



def get_depth_label_list(depth_init, depth_scale_array=np.linspace(0.5,1.5,32)):
    """Returns a list of tensor: depth label (previous_depth*depth_scale)
    
    depth_init: Tensor
    
    depth_scale_array: numpy array
    """
    depth_i_list = []
    depth_shape = depth_init.get_shape()
    if isinstance(depth_scale_array,(np.ndarray,)):
        for depth_step in depth_scale_array:
            depth_scale_tensor = tf.constant(depth_step, shape=depth_shape,dtype=tf.float32)
            depth_i = depth_init*depth_scale_tensor
            depth_i_list.append(depth_i)
    return depth_i_list

def create_border_mask_for_image(radius, image, name=None):
    """Creates a mask image that excludes the image borders
    e.g.
    00000
    01110
    00000
    
    radius: int 
        border radius
    image: Tensor
    """
    with tf.name_scope(name, "createBorderMaskForImage", [image]) as scope:
        image = tf.convert_to_tensor(image, name='image')
        shape = image.get_shape().as_list()
        assert len(shape)==4
        assert shape[-1] > 2*radius and shape[-2] > 2*radius
        new_shape = [shape[0],1,shape[2]-2*radius,shape[3]-2*radius]
        return tf.pad(tf.ones(new_shape),[[0,0],[0,0],[radius,radius],[radius,radius]], mode='CONSTANT')


def create_depthsweep_images_tensor(image,  rotation, translation, intrinsics, depth_values, border_radius=1, name=None):
    """Create warped images tensor (N*D*C*H*W) with the depth values.
    image: Tensor
    rotation: Tensor
    translation: Tensor
    intrinsics: Tensor
    depth_values: list of float or Tensor with shape NCHW with inverse depth values
    border_radius: int
    
    Returns the tensor of warped images in NDCHW format with 
    D = number of depth labels
    C = image channels
    
    Returns 
    A tensor with a mask which is 1 if there is a valid pixel for all depth labels in NCHW format
    with C=1.
    A mask which indicates pixels where all warped images have a valid value.
    The tensor with the generated depth values per pixel
    
    """
    with tf.name_scope(name, "createDepthsweepImagesTensor", [image, rotation, translation, intrinsics]):
        image = tf.convert_to_tensor(image, name='image', dtype=tf.float32)
        rotation = tf.convert_to_tensor(rotation, name='rotation', dtype=tf.float32)
        translation = tf.convert_to_tensor(translation, name='translation', dtype=tf.float32)
        intrinsics = tf.convert_to_tensor(intrinsics, name='intrinsics', dtype=tf.float32)
        
        image.get_shape().with_rank(4)
        rotation.get_shape().with_rank(2)
        translation.get_shape().with_rank(2)
        intrinsics.get_shape().with_rank(2)
        
        if isinstance(depth_values,(list,tuple,np.ndarray)):
            shape = image.get_shape().as_list()
            shape[1] = 1
            depths = []
            for d in depth_values:
                depths.append(tf.constant(value=d, shape=shape, dtype=tf.float32))
            depths = tf.concat(depths, axis=1)
        else: # Tensor
            depths = depth_values
        depths_shape = depths.get_shape()
        depths_shape.with_rank(4)
        num_labels = depths_shape[1].value
        
        mask_orig = create_border_mask_for_image(border_radius, image)
        mask = tf.tile(tf.expand_dims(mask_orig, axis=1), [1,num_labels,1,1,1])
        
        image = tf.tile(tf.expand_dims(image, axis=1), [1,num_labels,1,1,1])
        rotation = tf.tile(tf.expand_dims(rotation, axis=1), [1,num_labels,1])
        translation = tf.tile(tf.expand_dims(translation, axis=1), [1,num_labels,1])
        intrinsics = tf.tile(tf.expand_dims(intrinsics, axis=1), [1,num_labels,1])


        image_shape_NDCHW = image.get_shape().as_list()
        image_shape_NCHW = list(image_shape_NDCHW[1:])
        image_shape_NCHW[0] *= image.get_shape()[0].value
        image = tf.reshape(image, image_shape_NCHW)
        
        mask_shape_NDCHW = mask.get_shape().as_list()
        mask_shape_NCHW = list(mask_shape_NDCHW[1:])
        mask_shape_NCHW[0] *= mask.get_shape()[0].value
        mask = tf.reshape(mask, mask_shape_NCHW)
        
        flows = sops.depth_to_flow(
            depth=depths, 
            intrinsics=intrinsics, 
            rotation=rotation, 
            translation=translation,
            inverse_depth=True,
            normalize_flow=False,
        )
        
        images_warped = sops.warp2d_tf(image, flows, normalized=False, border_mode='value')
        images_warped = tf.reshape(images_warped, image_shape_NDCHW)

        masks_warped = sops.warp2d_tf(mask, flows, normalized=False, border_mode='value')
        masks_warped = tf.reshape(masks_warped, mask_shape_NDCHW)
        masks_warped_all = mask_orig*tf.cast(tf.reduce_all(tf.not_equal(masks_warped,0.0),axis=1), dtype=tf.float32)
        
        return images_warped, masks_warped_all, depths
        


        
def compute_sad_volume_with_confidence(img0, warped_images_tensor, mask, channel_weights=None, patch_size=3, use_conv3d_NCDHW=True, name=None):
    """Computes the sum of absolute differences for img0 and a tensor with warped images
 
    img0: Tensor
        image in NCHW format
        
    warped_images_tensor: Tensor
        warped images in NDCHW format
        D = number of depth labels
        C = image channels
        
    mask: Tensor
        mask in NCHW format with C=1
        
    channel_weights: list of float
        Individual weighting factors for the image channels. Defaults to 
        [5/32, 16/32, 11/32] for 3 channel images and [1,..]/num_channels for channels != 3.
    
    patch_size: int
        The spatial patch size

    use_conv3d_NCDHW: bool
        If True use the faster NCDHW format for tf.nn.conv3d
    """
    with tf.name_scope(name, "computeSADVolumeWithConfidence", [img0, warped_images_tensor]):
        img0 = tf.convert_to_tensor(img0, name='img0', dtype=tf.float32)
        warped_images_tensor = tf.convert_to_tensor(warped_images_tensor, name='warped_images_tensor', dtype=tf.float32)
        
        img0.get_shape().with_rank(4)
        warped_images_tensor.get_shape().with_rank(5)
        img0.get_shape()[1:].merge_with(warped_images_tensor.get_shape()[2:])
        
        num_channels = img0.get_shape()[1].value
        
        if channel_weights is None and num_channels==3:
            channel_weights = [5/32, 16/32, 11/32]
        elif channel_weights is None:
            channel_weights = np.ones((num_channels,),dtype=np.float32)/num_channels
        
        assert len(channel_weights) == num_channels
        
        img0_NDCHW = tf.expand_dims(img0, axis=1)
        abs_diff = tf.abs(img0_NDCHW-warped_images_tensor)
        
        if use_conv3d_NCDHW:
            kernel1 = tf.constant(channel_weights, dtype=tf.float32, shape=[1,1,1,3,1])
            sum1 = tf.nn.conv3d(tf.transpose(abs_diff,[0,2,1,3,4]), filter=kernel1, strides=[1,1,1,1,1], padding='SAME', data_format='NCDHW')

            kernel2 = tf.ones([1,patch_size,patch_size,1,1])/(patch_size**2)
            sum2 = tf.nn.conv3d(sum1, filter=kernel2, strides=[1,1,1,1,1], padding='SAME', data_format='NCDHW')
            sad_volume = tf.squeeze(sum2, axis=1)
        else:
            kernel1 = tf.constant(channel_weights, dtype=tf.float32, shape=[1,1,1,3,1])
            sum1 = tf.nn.conv3d(tf.transpose(abs_diff,[0,1,3,4,2]), filter=kernel1, strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC')

            kernel2 = tf.ones([1,patch_size,patch_size,1,1])/(patch_size**2)
            sum2 = tf.nn.conv3d(sum1, filter=kernel2, strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC')
            sad_volume = tf.squeeze(sum2, axis=-1)
        return sad_volume, mask*compute_confidence_for_costvolume_2(sad_volume)
    
    
def compute_sad_volume_for_sequence(img0, images, rotations, translations, intrinsics, depth_values, channel_weights=None, patch_size=3, sad_shift=None, name=None):
    """Computes the confidence weighted sum of SAD cost volumes between img0 and the given images
    
    img0: Tensor
        image in NCHW format
    
    images: list of Tensor
        List of images in NCHW format
        
    rotations: list of Tensor
        rotations in 3d angle axis format for each image in 'images'
        
    translations: list Tensor
        translations for each image in 'images'
        
    intrinsics: Tensor
        Intrinsic parameters valid for all images and img0
        
    depth_values: list of float or Tensor
        Either a list of inverse depth values or
        a tensor with shape NCHW
        
    channel_weights: list of float
        Individual weighting factors for the image channels. Defaults to 
        [5/32, 16/32, 11/32] for 3 channel images and [1,..]/num_channels for channels != 3.
    
    patch_size: int
        The spatial patch size

    sad_shift: float
        Shift the valid sad values by this value

    """
    with tf.name_scope(name, "computeSADVolumeForSequence", [img0,intrinsics]+images+rotations+translations):
        img0 = tf.convert_to_tensor(img0, name='img0', dtype=tf.float32)
        images = [tf.convert_to_tensor(v, name='images{0}'.format(i), dtype=np.float32) for i,v in enumerate(images)]
        rotations = [tf.convert_to_tensor(v, name='rotations{0}'.format(i), dtype=np.float32) for i,v in enumerate(rotations)]
        translations = [tf.convert_to_tensor(v, name='translations{0}'.format(i), dtype=np.float32) for i,v in enumerate(translations)]
        intrinsics = tf.convert_to_tensor(intrinsics, name='img0', dtype=tf.float32)
        
        assert len(images) == len(rotations)
        assert len(images) == len(translations)
        assert not isinstance(intrinsics,(list,tuple))
        
        border_radius = patch_size//2 + 1
        
        cv_list = []
        conf_list = []
        depths = depth_values
        for i in range(len(images)):
            image = images[i]
            rotation = rotations[i]
            translation = translations[i]
            
            warped, mask, depths = create_depthsweep_images_tensor(
                image=image,
                rotation=rotation,
                translation=translation, 
                intrinsics=intrinsics, 
                depth_values=depths, 
                border_radius=border_radius, 
                )
            cv, conf = compute_sad_volume_with_confidence(img0, warped, mask, channel_weights=channel_weights, patch_size=patch_size)
            cv_list.append(cv)
            conf_list.append(conf)
        
        if sad_shift is None:
            multiplied_cv = [cv_list[i]*conf_list[i] for i in range(len(cv_list))]
        else:
            multiplied_cv = [(cv_list[i]+sad_shift)*conf_list[i] for i in range(len(cv_list))]
        conf_sum = tf.add_n(conf_list)
        cv = sops.replace_nonfinite(tf.add_n(multiplied_cv)/conf_sum)
        
        return cv, conf_sum
            


