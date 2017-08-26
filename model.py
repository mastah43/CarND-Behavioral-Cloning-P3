from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing import image as keras_image
from keras.callbacks import CSVLogger
#import matplotlib.pyplot as plt #TODO
import pandas as pd
import numpy as np
import logging
import datetime
import math
from tqdm import tqdm

timestamp_start = datetime.datetime.now().strftime("%Y%m%d-%H%M")

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
        self.angles = []
        self.images = []


def create_model_nvidia():
    '''
    Creates a model based on the paper https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    by nvidia.
    :return:
    '''

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # TODO resize image to 45, 160, 3
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(24, 5, strides=2, name='conv_1', activation='elu'))
    model.add(Conv2D(36, 5, strides=2, name='conv_2', activation='elu'))
    model.add(Conv2D(48, 5, strides=2, name='conv_3', activation='elu'))
    model.add(Conv2D(64, 3, strides=1, name='conv_4', activation='elu'))
    model.add(Conv2D(64, 3, strides=1, name='conv_5', activation='elu'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(1164, activation='elu', name='dense1'))
    model.add(Dropout(0.5))

    model.add(Dense(100, activation='elu', name='dense2'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu', name='dense3'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu', name='dense5'))

    model.add(Dense(1, name='angle'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model


def load_dataset(dataset, drivelog_csv_path, header=None):
    def load_img(path):
        path = drivelog_csv_path[0:drivelog_csv_path.rfind('/')] + '/IMG/' + path.split('/')[-1]
        img = keras_image.load_img(path)
        img = keras_image.img_to_array(img)
        return img

    logger.info('loading drive log %s', drivelog_csv_path)
    drive_log = pd.read_csv(drivelog_csv_path,
                            header=header,
                            names=['img_path_center', 'img_path_left', 'img_path_right',
                                   'angle', 'throttle', 'break', 'speed'],
                            dtype={'angle':np.float32})
    logger.info('drive log size: %d', drive_log.size)

    len_dataset = drive_log.size

    # TODO use np memory map to deal with too low main mem?, see https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion/20664
    # correction angle for left and right camera image interpretes to 3°
    angle_correction = 0.12

    for index, drive_log_row in tqdm(drive_log[0:len_dataset].iterrows(), 'loading and augmenting training images'):

        def add_entry(angle, img):
            dataset.angles.append(angle)
            dataset.images.append(img)
            dataset.angles.append(angle * -1.0)
            dataset.images.append(np.fliplr(img))

        angle = drive_log_row['angle']
        # angle is in range -1 to 1 which interpretes to -25° to +25°
        add_entry(angle, load_img(drive_log_row['img_path_center']))
        add_entry(angle + angle_correction, load_img(drive_log_row['img_path_left']))
        add_entry(angle - angle_correction, load_img(drive_log_row['img_path_right']))

    return dataset


def train_model(model, dataset):
    angles = np.array(dataset.angles)
    images = np.array(dataset.images)

    # TODO use generator model.fit_generator, see https://keras.io/models/model/

    csv_logger = CSVLogger('training-history-' + timestamp_start + '.csv')
    train_history = model.fit(images, angles, epochs=7, validation_split=0.2, shuffle=True, callbacks=[csv_logger])
    # TODO increase epochs

    # TODO
    #plt.plot(train_history.history['loss'])
    #plt.plot(train_history.history['val_loss'])
    #plt.title('model mean squared error loss')
    #plt.ylabel('mean squared error loss')
    #plt.xlabel('epoch')
    #plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.figure().savefig('training-history-' + timestamp_start + '.png')

    return


def save_model(model, filename):
    model.save(filename)
    logger.info('saved model to %s', filename)


if __name__ == '__main__':
    dataset = DataSet()
    driving_dataset = load_dataset(dataset, '../drivelog1/driving_log.csv')
    driving_dataset = load_dataset(dataset, '../drivelog2/driving_log.csv', header=0)
    steering_model = create_model_nvidia()
    train_model(steering_model, driving_dataset)
    save_model(steering_model, 'model-' + timestamp_start + '.h5')