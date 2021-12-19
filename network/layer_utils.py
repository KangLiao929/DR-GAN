import tensorflow as tf

from keras.models import Model
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers.merge import Add
from keras.utils import conv_utils
from keras.layers.core import Dropout


def res_block(input, filters, kernel_size = (3, 3), strides = (1, 1), use_dropout = False):
    x = ReflectionPadding2D((1, 1))(input)
    x = Conv2D(filters = filters,
               kernel_size = kernel_size,
               strides = strides,)(x)
    x = BatchNormalization()(x)
    
    merged = Add()([input, x])
    return merged
    

def spatial_reflection_2d_padding(x, padding = ((1, 1), (1, 1)), data_format = None):
    # Pad the 2nd and 3rd dimensions of a 4D tensor
    
    # error when !=
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format' + str(data_format))
        
    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]),
                   list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")
    
# TODO: Credits
class ReflectionPadding2D(Layer):
    # This layer can add rows and columns or zeros at each side of image.
    
    def __init__(self, padding = (1, 1), data_format = None, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('padding should have two elements.')
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                        '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('padding should be either an int or a tuple')
        self.input_spec = InputSpec(ndim = 4)
        
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0], input_shape[1], rows, cols)
            
        if self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0], rows, cols, input_shape[3])
            
            
    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs, 
                                             padding = self.padding, 
                                             data_format = self.data_format)
    
    def get_config(self):
        config = {'padding' : self.padding,
                  'data_format' : self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

            
            
        