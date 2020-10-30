import struct
import array
import numpy



DATA_TYPES = {0x08: 'B',  # unsigned byte
              0x09: 'b',  # signed byte
              0x0b: 'h',  # short (2 bytes)
              0x0c: 'i',  # int (4 bytes)
              0x0d: 'f',  # float (4 bytes)
              0x0e: 'd'}  # double (8 bytes)

FILE_NAMES = ['train-images.idx3-ubyte',
              'train-labels.idx1-ubyte',
              't10k-images.idx3-ubyte',
              't10k-labels.idx1-ubyte']

def mnist_data(i):
    filename = 'database//' + FILE_NAMES[i]
    fd = open(filename, 'rb')
    header = fd.read(4)
    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
    data_type = DATA_TYPES[data_type]
    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))
    data = array.array(data_type, fd.read())
    data.byteswap()
    
    return np.array(data).reshape(dimension_sizes)


TRAIN_IMAGES = mnist_data(0)
TRAIN_LABELS = mnist_data(1)
TEST_IMAGES = mnist_data(2)
TEST_LABELS = mnist_data(3)

LEN_TRAIN = len(TRAIN_LABELS)
LEN_TEST = len(TEST_LABELS)




