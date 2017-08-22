from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Flatten
from keras.layers.convolutional import Convolution2D
from keras.preprocessing import image
import pandas as pd
import numpy as np
import logging
import math
from tqdm import tqdm

# setup logging
logger = logging.getLogger('model')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class DataSet( object ):
   def __init__( self ):
       self.angles = None
       self.images = None

def create_model_transfered():

    def freeze_layers(model):
        for layer in model.layers:
            layer.trainable = False

    # TODO use googlenet since it is fast in inference so training
    # model = create_googlenet('googlenet_weights.h5')
    # TODO remove layers of googlenet: model.layers.pop()

    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    '''
    # TODO

    input = Input(shape=(224, 224, 3))
    model = VGG16(input_tensor=input, include_top=False)
    freeze_layers(model)

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    '''
    model.add(Dense(1, name='angle'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def create_model_trivial():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def create_model_nvidia():
    '''
    Creates a model based on the paper https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    by nvidia.
    :return:
    '''

    # TODO use YUV planes of image

    model = Sequential()
    # TODO normalization layer
    # TODO add dropout
    # TODO add L2 regularization
    #model.add(Flatten(input_shape=(160, 320, 3)))

    model.add(Convolution2D(24, 31, 98, input_shape=(160, 320, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 14, 47))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 22))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 20))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 1, 18))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1, name='angle'))
    # TODO model.add(Dropout(0.2))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def load_dataset(drivelog_csv_path):
    def load_img(path):
        path = drivelog_csv_path[0:drivelog_csv_path.rfind('/')] + '/IMG/' + path.split('/')[-1]
        img = image.load_img(path)
        img = image.img_to_array(img)
        return img

    logger.info("loading drive log %s", drivelog_csv_path)
    drive_log = pd.read_csv(drivelog_csv_path,
        names=['img_path_center', 'img_path_left', 'img_path_right', 'angle', 'throttle', 'break', 'speed'],
        dtype={'angle':np.float32})
    logger.info("drive log size: %d", drive_log.size)

    dataset = DataSet()
    len_dataset = drive_log.size
    angles = []
    images = []

    # TODO use np memory map to deal with too low main mem?, see https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/20664
    for index, drive_log_row in tqdm(drive_log[0:len_dataset].iterrows(), "loading and normalizing training images"):
        # TODO angle = float(drive_log_row['angle'])
        angle = drive_log_row['angle']
        angle_diff_correct = (math.pi/180) * 10
        angles.append(angle)
        images.append(load_img(drive_log_row['img_path_center']))

        #dataset.angles.append(load_img(drive_log_row['img_path_left']))
        #dataset.images.append(angle - angle_diff_correct)
        #dataset.angles.append(load_img(drive_log_row['img_path_right']))
        #dataset.images.append(angle + angle_diff_correct)

    dataset.angles = np.array(angles)
    dataset.images = np.array(images)

    return dataset


def augment_dataset(dataset):
    #data_gen = image.ImageDataGenerator(horizontal_flip=True)
    pass


def train_model(model, dataset):

    # TODO create bottleneck features to speed up training (so inference of transfer learning model is not needed during training)

    # TODO could use model.fit_generator to load and augment images while training on gpu
    model.fit(dataset.images, dataset.angles, epochs=1, validation_split=0.2, shuffle=True)
    # TODO validate model after each epoch and print out results
    pass


def save_model(model, filename):
    model.save(filename)
    logger.info("saved model to %s", filename)


img = None
if __name__ == '__main__':

    # TODO record new training data using mouse input for fine grained drive angles

    driving_dataset = load_dataset('../drivelog1/driving_log.csv')
    augment_dataset(driving_dataset)
    steering_model = create_model_nvidia()
    train_model(steering_model, driving_dataset)
    save_model(steering_model, 'model-1.h5')