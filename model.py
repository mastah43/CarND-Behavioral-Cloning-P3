from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing import image as keras_image
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import logging
import datetime
import argparse
import random
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
        df.set_value(index, 'img_path_center', map_img_path(row['img_path_center']))
        df.set_value(index, 'img_path_left', map_img_path(row['img_path_left']))
        df.set_value(index, 'img_path_right', map_img_path(row['img_path_right']))

    logger.info('drive log size: %d', len(df))
    return df


def train_model(model, drive_log):

    def sample_generator(drive_log, batch_size):
        num_samples = len(drive_log)

        # shuffle so that order of training samples must not be relevant for training outcome
        shuffle(drive_log, random_state=4711)

        # correction angle for left and right camera image; interpretes to 6°
        # angle is in range -1 to 1 which interpretes to -25° to +25°
        angle_correction = 0.24

        images = []
        angles = []

        def load_img(path):
            img = keras_image.load_img(path)
            img = keras_image.img_to_array(img)
            return img

        def add_sample(angle, img):
            angles.append(angle)
            images.append(img)

        def add_sample_and_augment(angle, img):
            add_sample(angle, img)
            add_sample(angle * -1.0, np.fliplr(img))

        while 1:
            for index, drive_log_row in drive_log.iterrows():
                angle_center = drive_log_row['angle']
                angle_left = angle_center + angle_correction
                angle_right = angle_center - angle_correction

                add_sample_and_augment(angle_center, load_img(drive_log_row['img_path_center']))
                add_sample_and_augment(angle_left, load_img(drive_log_row['img_path_left']))
                add_sample_and_augment(angle_right, load_img(drive_log_row['img_path_right']))

                if len(angles) >= batch_size:
                    yield shuffle(np.array(images), np.array(angles))
                    images, angles = [], []

    if limit_samples > 0:
        drive_log = drive_log.sample(limit_samples, random_state=4711)

    train_drive_log, validate_drive_log = train_test_split(drive_log, test_size=0.2)
    train_generator = sample_generator(train_drive_log, batch_size)
    validate_generator = sample_generator(validate_drive_log, batch_size)

    model_checkpoint = ModelCheckpoint(
        filepath='model-' + timestamp_start + '-{epoch:02d}-{val_loss:.4f}.h5',
        verbose=1,
        save_best_only=True)
    csv_logger = CSVLogger('training-history-' + timestamp_start + '.csv')

    augmentation_factor = 6
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=(train_drive_log.size/batch_size)*augmentation_factor,
                        validation_data=validate_generator,
                        validation_steps=(validate_drive_log.size/batch_size)*augmentation_factor,
                        epochs=epochs,
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


def init_params_from_cmd_args():
    parser = argparse.ArgumentParser(description="training model for udacity carnd project 3")
    parser.add_argument('--epochs', type=int, default=4, help='# of training epochs')
    parser.add_argument('--limit_samples', type=int, default=-1, help='# of samples to')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--batch_per_epoch', type=int, default=-1, help='batch per epoch')
    args = parser.parse_args()
    global epochs
    global limit_samples
    global batch_size
    epochs = args.epochs
    batch_size = args.epochs
    limit_samples = args.limit_samples


epochs = None
limit_samples = None
batch_size = None
batches_per_epoch = None


if __name__ == '__main__':

    # ensure same result for each training run on same training data and parameters
    random.seed(4711)

    init_params_from_cmd_args()

    drive_log1 = load_drive_log('../drivelog1/driving_log.csv')
    drive_log2 = load_drive_log('../drivelog2/driving_log.csv', header=0)
    drive_log_all = drive_log2 # TODO pd.concat([drive_log1, drive_log2])

    steering_model = create_model_nvidia()
    train_model(steering_model, drive_log_all)