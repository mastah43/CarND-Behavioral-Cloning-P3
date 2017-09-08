from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing import image as keras_image
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#import matplotlib.pyplot as plt #TODO
import pandas as pd
import numpy as np
import logging
import datetime
import argparse
import os

timestamp_start = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# setup logging
logger = logging.getLogger('model')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def create_model_nvidia():
    '''
    Creates a model based on the paper https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    by nvidia.
    :return:
    '''

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # TODO resize image to 45, 160, 3 ?
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


def load_drive_log(csv_path, header=None):
    logger.info('loading drive log %s', csv_path)
    df = pd.read_csv(csv_path,
                     header=header,
                     names=['img_path_center', 'img_path_left', 'img_path_right',
                            'angle', 'throttle', 'break', 'speed'],
                     dtype={'angle':np.float32})

    def map_img_path(path):
        return os.path.join(os.path.dirname(os.path.realpath(csv_path)), 'IMG', path.split('/')[-1])

    for index,row in df.iterrows():
        # TODO improve efficiency
        df.set_value(index, 'img_path_center', map_img_path(row['img_path_center']))
        df.set_value(index, 'img_path_left', map_img_path(row['img_path_left']))
        df.set_value(index, 'img_path_right', map_img_path(row['img_path_right']))

    logger.info('drive log size: %d', df.size)
    return df


def train_model(model, drive_log):

    def sample_generator(drive_log, batch_size=128):
        num_samples = len(drive_log)
        shuffle(drive_log)
        # correction angle for left and right camera image; interpretes to 6°
        angle_correction = 0.24

        # TODO create drive_log with augmented entries (using python function for augmentation)

        while 1:
            for offset in range(0, num_samples, batch_size):
                batch_samples = drive_log[offset:offset + batch_size]
                images = []
                angles = []

                def load_img(path):
                    img = keras_image.load_img(path)
                    img = keras_image.img_to_array(img)
                    return img

                def add_sample_and_augment(angle, img):
                    angles.append(angle)
                    images.append(img)
                    angles.append(angle * -1.0)
                    images.append(np.fliplr(img))

                for index, drive_log_row in batch_samples.iterrows():
                    # angle is in range -1 to 1 which interpretes to -25° to +25°
                    angle_center = drive_log_row['angle']
                    add_sample_and_augment(angle_center, load_img(drive_log_row['img_path_center']))
                    angle_left = angle_center + angle_correction
                    angle_right = angle_center - angle_correction
                    add_sample_and_augment(angle_left, load_img(drive_log_row['img_path_left']))
                    add_sample_and_augment(angle_right, load_img(drive_log_row['img_path_right']))

                yield shuffle(np.array(images), np.array(angles))

    train_drive_log, validate_drive_log = train_test_split(drive_log, test_size=0.2)
    batch_size = 64
    train_generator = sample_generator(train_drive_log, batch_size)
    validate_generator = sample_generator(validate_drive_log, batch_size)

    model_checkpoint = ModelCheckpoint(
        filepath='model-' + timestamp_start + '-{epoch:02d}-{val_loss:.4f}.h5',
        verbose=1,
        save_best_only=True)
    csv_logger = CSVLogger('training-history-' + timestamp_start + '.csv')
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_drive_log.size/batch_size,
                        validation_data=validate_generator,
                        validation_steps=validate_drive_log.size/batch_size,
                        nb_epoch=7,
                        callbacks=[csv_logger, model_checkpoint])

    # TODO plot history of losses
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
    # TODO batchsize arg
    #parser = argparse.ArgumentParser(description='Process some integers.')
    #parser.add_argument('--batchsize', dest='accumulate', action='store_const',
    #                    const=sum, default=max,
    #                    help='sum the integers (default: find the max)')
    #args = parser.parse_args()

    drive_log1 = load_drive_log('../drivelog1/driving_log.csv')
    drive_log2 = load_drive_log('../drivelog2/driving_log.csv', header=0)
    drive_log_all = pd.concat([drive_log1, drive_log2])

    steering_model = create_model_nvidia()
    train_model(steering_model, drive_log_all)
    save_model(steering_model, 'model-' + timestamp_start + '.h5')