from .helpers import *
import tensorflow as tf

def depth_fb_block(image, 
                   stop_direct_gradients=False, 
                   weights_regularizer=None,
                   stop_input_gradients=True):
    """Creates a fixed band depth network
    
    image: Tensor
        The tensor format is NCHW with C == 3.
        
    stop_direct_gradients: bool
        If True do not compute a gradient for the direct connections

    weights_regularizer: function
        A function returning a weight regularizer
    
    stop_input_gradients: bool
        If True do not back propogate throught the input image
    """
    conv_params = {'kernel_regularizer': weights_regularizer, 'data_format': 'channels_first'}
    fc_params = {'kernel_regularizer': weights_regularizer}
    
    with tf.variable_scope('depth_encoder',reuse=tf.AUTO_REUSE):
        if stop_input_gradients:
            image = tf.stop_gradient(image)
        
        # depth part
        conv0 = convrelu2(name='conv0', inputs=image, num_outputs=(48,48), kernel_size=7, stride=1, **conv_params)
        conv0_1 = convrelu2(name='conv0_1', inputs=conv0, num_outputs=48, kernel_size=3, stride=1, **conv_params)
        
        conv1 = convrelu2(name='conv1', inputs=conv0_1, num_outputs=(64,64), kernel_size=7, stride=2, **conv_params)
        conv1_1 = convrelu2(name='conv1_1', inputs=conv1, num_outputs=64, kernel_size=3, stride=1, **conv_params)
        
        conv2 = convrelu2(name='conv2', inputs=conv1_1, num_outputs=(128,128), kernel_size=5, stride=2, **conv_params)
        conv2_1 = convrelu2(name='conv2_1', inputs=conv2, num_outputs=128, kernel_size=3, stride=1, **conv_params)
        
        conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(256,256), kernel_size=5, stride=2, **conv_params)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=256, kernel_size=3, stride=1, **conv_params)
        
        conv4 = convrelu2(name='conv4', inputs=conv3_1, num_outputs=(512,512), kernel_size=3, stride=2, **conv_params)
        conv4_1 = convrelu2(name='conv4_1', inputs=conv4, num_outputs=512, kernel_size=3, stride=1, **conv_params)
    
        print(conv0, conv0_1)
        print(conv1, conv1_1)
        print(conv2, conv2_1)
        print(conv3, conv3_1)
        print(conv4, conv4_1)
        
    with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
        # expanding part
        with tf.variable_scope('refine4'):
            concat4 = _refine(
                inp=conv4_1, 
                num_outputs=256, 
                features_direct=tf.stop_gradient(conv3_1) if stop_direct_gradients else conv3_1,
                data_format='channels_first',
            )
        print(concat4)

        with tf.variable_scope('refine3'):
            #128+128+128
            concat3 = _refine(
                inp=concat4, 
                num_outputs=128, 
                #features_direct=tf.stop_gradient(conv2_1) if stop_direct_gradients else conv2_1,
                #upsampled_prediction = conv2_1_cv,
                data_format='channels_first',
            )
            #concat3_1 = concat3 + conv2_1
            print(concat3)

        with tf.variable_scope('refine2'):
            #64+64+64
            concat2 = _refine(
                inp=concat3, 
                num_outputs=64, 
                #features_direct=tf.stop_gradient(conv1_1) if stop_direct_gradients else conv1_1,
                #upsampled_prediction=conv1_1_cv,
                data_format='channels_first',
            )
            #concat2_1 = concat2 + conv1_1
            print(concat2)
            
        with tf.variable_scope('refine1'):
            #32+32+32
            concat1 = _refine(
                inp=concat2, 
                num_outputs=48, 
                #features_direct=tf.stop_gradient(conv0_1) if stop_direct_gradients else conv0_1,
                #upsampled_prediction=conv0_1_cv,
                data_format='channels_first',
            )
            #concat1_1 = concat1 + conv0_1
            print(concat1)
            
            costvolume = convrelu2(name='costvolume', inputs=concat1, num_outputs=32, kernel_size=3, stride=1, **conv_params)
            
            print(costvolume)

            predictions = _predict_depth(costvolume,normal=False,**conv_params)

    return {
        'predict_depth0': predictions,
        }



def depth_nb_block(image,
                   cv_raw,
                   stop_direct_gradients=False,
                   weights_regularizer=None,
                   stop_input_gradients=True):
    """Creates a fixed band depth network
    
    image: Tensor
        Keyframe image. The tensor format is NCHW with C == 3.
        
    cv_raw: Tensor
        Computed cost volume. The tensor format is NCHW with C == 32.
        
    stop_direct_gradients: bool
        If True do not compute a gradient for the direct connections

    weights_regularizer: function
        A function returning a weight regularizer
    
    stop_input_gradients: bool
        If True do not back propogate throught the input image
    """
    conv_params = {'kernel_regularizer': weights_regularizer, 'data_format': 'channels_first'}
    fc_params = {'kernel_regularizer': weights_regularizer}
    
    with tf.variable_scope('depth_encoder',reuse=tf.AUTO_REUSE):
        if stop_input_gradients:
            image = tf.stop_gradient(image)
            cv_raw = tf.stop_gradient(cv_raw)
                        
        # depth part
        conv0 = convrelu2(name='conv0', inputs=image, num_outputs=(32,32), kernel_size=7, stride=1, **conv_params)
        conv0_1 = convrelu2(name='conv0_1', inputs=conv0, num_outputs=32, kernel_size=3, stride=1, **conv_params)
        
        conv1 = convrelu2(name='conv1', inputs=conv0_1, num_outputs=(64,64), kernel_size=7, stride=2, **conv_params)
        conv1_1 = convrelu2(name='conv1_1', inputs=conv1, num_outputs=64, kernel_size=3, stride=1, **conv_params)
        
        conv2 = convrelu2(name='conv2', inputs=conv1_1, num_outputs=(128,128), kernel_size=5, stride=2, **conv_params)
        conv2_1 = convrelu2(name='conv2_1', inputs=conv2, num_outputs=128, kernel_size=3, stride=1, **conv_params)
        
        conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(128,128), kernel_size=5, stride=2, **conv_params)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=128, kernel_size=3, stride=1, **conv_params)
        
        conv4 = convrelu2(name='conv4', inputs=conv3_1, num_outputs=(256,256), kernel_size=3, stride=2, **conv_params)
        conv4_1 = convrelu2(name='conv4_1', inputs=conv4, num_outputs=256, kernel_size=3, stride=1, **conv_params)
    
        print(conv0, conv0_1)
        print(conv1, conv1_1)
        print(conv2, conv2_1)
        print(conv3, conv3_1)
        print(conv4, conv4_1)
        
    with tf.variable_scope('cv_encoder',reuse=tf.AUTO_REUSE):  
        # raw cv part
        conv0_cv = convrelu2(name='conv0', inputs=cv_raw, num_outputs=(32,32), kernel_size=3, stride=1, **conv_params)
        conv0_1_cv = convrelu2(name='conv0_1', inputs=conv0, num_outputs=32, kernel_size=3, stride=1, **conv_params)
        
        conv1_cv = convrelu2(name='conv1', inputs=conv0_1_cv, num_outputs=(64,64), kernel_size=3, stride=2, **conv_params)
        conv1_1_cv = convrelu2(name='conv1_1', inputs=conv1_cv, num_outputs=64, kernel_size=3, stride=1, **conv_params)
        
        conv2_cv = convrelu2(name='conv2', inputs=conv1_1_cv, num_outputs=(128,128), kernel_size=3, stride=2, **conv_params)
        conv2_1_cv = convrelu2(name='conv2_1', inputs=conv2_cv, num_outputs=128, kernel_size=3, stride=1, **conv_params)
        
    
        print(conv0_cv, conv0_1_cv)
        print(conv1_cv, conv1_1_cv)
        print(conv2_cv, conv2_1_cv)

        
    with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
        # expanding part
        with tf.variable_scope('refine4'):
            concat4 = _refine(
                inp=conv4_1, 
                num_outputs=128, 
                features_direct=tf.stop_gradient(conv3_1) if stop_direct_gradients else conv3_1,
                data_format='channels_first',
            )
        print(concat4)

        with tf.variable_scope('refine3'):
            #128+128+128
            concat3 = _refine(
                inp=concat4, 
                num_outputs=128, 
                #features_direct=tf.stop_gradient(conv2_1) if stop_direct_gradients else conv2_1,
                #upsampled_prediction = conv2_1_cv,
                data_format='channels_first',
            )
            concat3_1 = concat3 + conv2_1 + conv2_1_cv
            print(concat3)

        with tf.variable_scope('refine2'):
            #64+64+64
            concat2 = _refine(
                inp=concat3_1, 
                num_outputs=64, 
                #features_direct=tf.stop_gradient(conv1_1) if stop_direct_gradients else conv1_1,
                #upsampled_prediction=conv1_1_cv,
                data_format='channels_first',
            )
            concat2_1 = concat2 + conv1_1 + conv1_1_cv
            print(concat2)
            
        with tf.variable_scope('refine1'):
            #32+32+32
            concat1 = _refine(
                inp=concat2_1, 
                num_outputs=32, 
                #features_direct=tf.stop_gradient(conv0_1) if stop_direct_gradients else conv0_1,
                #upsampled_prediction=conv0_1_cv,
                data_format='channels_first',
            )
            concat1_1 = concat1 + conv0_1 + conv0_1_cv
            print(concat1)
            
            costvolume = convrelu2(name='costvolume', inputs=concat1_1, num_outputs=32, kernel_size=3, stride=1, **conv_params)
            
            print(costvolume)

    return costvolume

def depth_nb_refine_block(block_inputs,
                          stop_direct_gradients=False,
                          weights_regularizer=None,
                          depth_features=None,
                          pyramid_pooling=False):
    """Creates a narrow band refinement depth network
    
    block_inputs: Tensor
        
    stop_direct_gradients: bool
        If True do not compute a gradient for the direct connections

    weights_regularizer: function
        A function returning a weight regularizer
    
    Returns a tuple with the optical flow and confidence at resolution 
    level 5 and 2.
    """
    conv_params = {'kernel_regularizer': weights_regularizer, 'data_format': 'channels_first'}
    fc_params = {'kernel_regularizer': weights_regularizer}
    
    with tf.variable_scope('depth_org_reso',reuse=tf.AUTO_REUSE):
        # contracting part
        conv0 = convrelu2(name='conv0', inputs=block_inputs, num_outputs=(32,32), kernel_size=9, stride=1, **conv_params)
        conv0_1 = convrelu2(name='conv0_1', inputs=conv0, num_outputs=32, kernel_size=3, stride=1, **conv_params)
        
        conv1 = convrelu2(name='conv1', inputs=conv0_1, num_outputs=(64,64), kernel_size=9, stride=2, **conv_params)
        conv1_1 = convrelu2(name='conv1_1', inputs=conv1, num_outputs=64, kernel_size=3, stride=1, **conv_params)
        
        conv2 = convrelu2(name='conv2', inputs=conv1_1, num_outputs=(128,128), kernel_size=7, stride=2, **conv_params)
        conv2_1 = convrelu2(name='conv2_1', inputs=conv2, num_outputs=128, kernel_size=3, stride=1, **conv_params)
        
        conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(256,256), kernel_size=5, stride=2, **conv_params)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=256, kernel_size=3, stride=1, **conv_params)
        
        conv4 = convrelu2(name='conv4', inputs=conv3_1, num_outputs=(512,512), kernel_size=5, stride=2, **conv_params)
        conv4_1 = convrelu2(name='conv4_1', inputs=conv4, num_outputs=512, kernel_size=3, stride=1, **conv_params)
    
        print(conv0, conv0_1)
        print(conv1, conv1_1)
        print(conv2, conv2_1)
        print(conv3, conv3_1)
        print(conv4, conv4_1)

        if depth_features is not None:
            features_concat = tf.concat([conv4_1, depth_features], axis=1)
            features = convrelu2(name='conv_features', inputs=features_concat, num_outputs=512, kernel_size=3, stride=1, **conv_params)
            
        else:
            features = conv4_1
        # expanding part
        with tf.variable_scope('refine4'):
            concat4 = _refine(
                inp=features, 
                num_outputs=256, 
                features_direct=tf.stop_gradient(conv3_1) if stop_direct_gradients else conv3_1,
            )
            print(concat4)

        with tf.variable_scope('refine3'):
            concat3 = _refine(
                inp=concat4, 
                num_outputs=128, 
                features_direct=tf.stop_gradient(conv2_1) if stop_direct_gradients else conv2_1,
            )
            print(concat3)

        with tf.variable_scope('refine2'):
            concat2 = _refine(
                inp=concat3, 
                num_outputs=64, 
                features_direct=tf.stop_gradient(conv1_1) if stop_direct_gradients else conv1_1,
            )
            print(concat2)
            
        with tf.variable_scope('refine1'):
            concat1_tmp = _refine(
                inp=concat2, 
                num_outputs=32, 
                features_direct=tf.stop_gradient(conv0_1) if stop_direct_gradients else conv0_1,
            )
            
            if pyramid_pooling:
                concat1 = pyramid_pooling_module(concat1_tmp, 4)
            else:
                concat1 = concat1_tmp
            print(concat1)
        with tf.variable_scope('predict_depth0'):
            predict_depth0 = _predict_depth(concat1, **conv_params)
            print(predict_depth0)
    return { 
        'predict_depth0': predict_depth0,
        'depth_features': features,
        }



def _predict_depth(inp, normal=False, **kwargs):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor


    weights_regularizer: function
        A function returning a 
    """

    tmp = convrelu(
        inputs=inp,
        num_outputs=24,
        kernel_size=3,
        strides=1,
        name="conv1",
        **kwargs,
    )
    predicted_depth = conv2d(
        inputs=tmp,
        num_outputs=4 if normal else 1, # generate 1 depth and 3 normal channels if normal
        kernel_size=3,
        strides=1,
        name="conv2",
        **kwargs,
    )
    return predicted_depth

def _predict_flow(inp, predict_confidence=False, weights_regularizer=None, data_format='NCHW'):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor

    predict_confidence: bool
        If True the output tensor has 4 channels instead of 2.
        The last two channels are the x and y flow confidence.

    weights_regularizer: function
        A function returning a 
    """

    tmp = tf.contrib.layers.convolution2d(
        inputs=inp,
        num_outputs=24,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=myLeakyRelu,
        weights_initializer=weights_initializer(),
        weights_regularizer=weights_regularizer,
        data_format=data_format,
        scope="conv1",
    )
    
    output = tf.contrib.layers.convolution2d(
        inputs=tmp,
        num_outputs=4 if predict_confidence else 2,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=None,
        weights_initializer=weights_initializer(),
        weights_regularizer=weights_regularizer,
        data_format=data_format,
        scope="conv2",
    )
    
    return output
def _upsample_prediction(inp, num_outputs, data_format='NCHW'):
    """Upconvolution for upsampling predictions
    
    inp: Tensor 
        Tensor with the prediction
        
    num_outputs: int
        Number of output channels. 
        Usually this should match the number of channels in the predictions
    """
    output = tf.contrib.layers.convolution2d_transpose(
        inputs=inp,
        num_outputs=num_outputs,
        kernel_size=4,
        stride=2,
        padding='SAME',
        activation_fn=None,
        weights_initializer=weights_initializer(),
        weights_regularizer=None,
        data_format=data_format,
        scope="upconv",
    )
    return output

def _refine(inp, num_outputs, upsampled_prediction=None, features_direct=None, data_format='channels_first', **kwargs):
    """ Generates the concatenation of 
         - the previous features used to compute the flow/depth
         - the upsampled previous flow/depth
         - the direct features that already have the correct resolution

    inp: Tensor
        The features that have been used before to compute flow/depth

    num_outputs: int 
        number of outputs for the upconvolution of 'features'

    upsampled_prediction: Tensor
        The upsampled flow/depth prediction

    features_direct: Tensor
        The direct features which already have the correct resolution
    """

    upsampled_features = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name="upconv",
        **kwargs,
    )
    inputs = [upsampled_features, upsampled_prediction, features_direct]
    concat_inputs = [ x for x in inputs if not x is None ]
    
    if data_format == 'channels_first':
        return tf.concat(concat_inputs, axis=1)
    else: # NHWC
        return tf.concat(concat_inputs, axis=3)
    


