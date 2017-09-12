from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
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
from PIL import Image


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")

# setup logging
logger = logging.getLogger('model')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

timestamp_start = get_timestamp()


def create_model_nvidia():
    '''
    Creates a model based on the paper https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    by nvidia.
    :return:
    '''

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
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

    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['mse'])

    return model


def load_drive_log(csv_path, header=None):
    def map_img_path(path):
        # support also driving logs recorded on windows
        path = path.replace('\\', '/')
        return os.path.join(os.path.dirname(os.path.realpath(csv_path)), 'IMG', path.split('/')[-1])

    logger.info('loading drive log %s', csv_path)
    df = pd.read_csv(csv_path,
                     header=header,
                     names=['img_path_center', 'img_path_left', 'img_path_right',
                            'angle', 'throttle', 'break', 'speed'],
                     dtype={'angle': np.float32})

    # correct image paths
    for index, row in df.iterrows():
        df.set_value(index, 'img_path_center', map_img_path(row['img_path_center']))
        df.set_value(index, 'img_path_left', map_img_path(row['img_path_left']))
        df.set_value(index, 'img_path_right', map_img_path(row['img_path_right']))

    len_loaded = len(df)

    # remove samples not moving
    df = df[(df.speed > 0.1)]
    logger.info('removed %d samples where car was not moving', (len(df) - len_loaded))

    logger.info('drive log size: %d', len(df))
    return df


def plot_train_samples_to_file(images, angles, filename):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for j, (img, angle) in enumerate(zip(images, angles)):
        plt.subplot(6, 5, j + 1)
        plt.title("angle=%.4f" % angle)
        plt.imshow(img)
    fig.set_size_inches(30, 15)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    logger.info('plotted images to ' + filename)


def train_model(model, drive_log):
    def sample_generator(drive_log, batch_size, plot_samples_count=0):
        num_samples = len(drive_log)

        # correction angle for left and right camera image; interpretes to 6°
        # angle is in range -1 to 1 which interpretes to -25° to +25°
        angle_correction = 0.24

        images = []
        angles = []

        def load_img(path):
            return np.asarray(Image.open(path))

        def add_sample(angle, img):
            angles.append(angle)
            images.append(img)

        def add_sample_and_flip(angle, img):
            add_sample(angle, img)
            add_sample(angle * -1.0, np.fliplr(img))

        def image_viewpoint_transform(im, isdeg=True):
            import cv2

            # Viewpoint transform for recovery
            theta = random.randint(-80, 80)
            if isdeg:
                theta = np.deg2rad(theta)

            f = 2.
            h, w, _ = im.shape
            cx = cz = np.cos(0)
            sx = sz = np.sin(0)
            cy = np.cos(theta)
            sy = np.sin(theta)

            R = np.array([[cz * cy, cz * sy * sx - sz * cx], [sz * cy, sz * sy * sx + cz * cx], [-sy, cy * sx]],
                         np.float32)

            pts1 = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]

            pts2 = []
            mx, my = 0, 0
            for i in range(4):
                pz = pts1[i][0] * R[2][0] + pts1[i][1] * R[2][1]
                px = w / 2 + (pts1[i][0] * R[0][0] + pts1[i][1] * R[0][1]) * f * h / (f * h + pz)
                py = h / 2 + (pts1[i][0] * R[1][0] + pts1[i][1] * R[1][1]) * f * h / (f * h + pz)
                pts2.append([px, py])

            pts2 = np.array(pts2, np.float32)
            pts1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

            x1, x2 = int(min(pts2[0][0], pts2[3][0])), int(max(pts2[1][0], pts2[2][0]))
            y1, y2 = int(max(pts2[0][1], pts2[1][1])), int(min(pts2[2][1], pts2[3][1]))

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(im, M, (w, h), cv2.INTER_NEAREST | cv2.INTER_NEAREST)

            x1 = np.clip(x1, 0, w)
            x2 = np.clip(x2, 0, w)
            y1 = np.clip(y1, 0, h)
            y2 = np.clip(y2, 0, h)
            z = dst[y1:y2, x1:x2]
            x, y, _ = z.shape
            if x == 0 or y == 0:
                return
            return cv2.resize(z, (w, h), interpolation=cv2.INTER_AREA), -np.rad2deg(theta) / 200.

        batch_index = 0
        while 1:
            for index, drive_log_row in drive_log.iterrows():
                angle_center = drive_log_row['angle']
                angle_left = angle_center + angle_correction
                angle_right = angle_center - angle_correction

                add_sample_and_flip(angle_center, load_img(drive_log_row['img_path_center']))
                add_sample_and_flip(angle_left, load_img(drive_log_row['img_path_left']))
                add_sample_and_flip(angle_right, load_img(drive_log_row['img_path_right']))

                """
                # TODO augment using perspective transformation and shifting
                #img_center = keras_image.load_img(drive_log_row['img_path_center'])
                img_center = load_img(drive_log_row['img_path_center'])
                #img_center_warped, angle_adj = img_center, 0.
                img_center_warped, angle_adj = image_viewpoint_transform(img_center)
                add_sample(angle_center + angle_adj, img_center_warped)
                """

                if len(angles) >= batch_size:
                    if plot_samples_count > 0 and batch_index == 0:
                        plot_samples_count = min(plot_samples_count, len(images))
                        plot_train_samples_to_file(images[0:plot_samples_count], angles[0:plot_samples_count],
                                                   'train-samples-excerpt-' + get_timestamp() + '.png')

                    yield shuffle(np.array(images), np.array(angles))
                    batch_index += 1
                    images, angles = [], []

    # shuffle so that order of training samples must not be relevant for training outcome
    shuffle(drive_log, random_state=4711)
    if limit_samples > 0:
        drive_log = drive_log.sample(limit_samples, random_state=4711)
        logger.info('limited data set to: %d', len(drive_log))

    train_drive_log, validate_drive_log = train_test_split(drive_log, test_size=0.2)
    train_generator = sample_generator(train_drive_log, batch_size, plot_samples_count=plot_train_samples_count)
    validate_generator = sample_generator(validate_drive_log, batch_size)

    model_checkpoint = ModelCheckpoint(
        filepath='model-' + timestamp_start + '-{epoch:02d}-{val_loss:.4f}.h5',
        verbose=1,
        save_best_only=True)
    csv_logger = CSVLogger('training-history-' + timestamp_start + '.csv')

    augmentation_factor = 6
    steps_per_epoch = train_drive_log.size * augmentation_factor * 1.0 / batch_size
    validation_steps = validate_drive_log.size * augmentation_factor * 1.0 / batch_size
    logger.info('using batch_size: %d', batch_size)
    logger.info('using steps per epoch: %d', steps_per_epoch)
    logger.info('using validation steps: %d', validation_steps)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validate_generator,
                        validation_steps=validation_steps,
                        epochs=epochs,
                        callbacks=[csv_logger, model_checkpoint])

    return


epochs = None
limit_samples = None
batch_size = None
batches_per_epoch = None
plot_train_samples_count = 0

if __name__ == '__main__':
    # ensure same result for each training run on same training data and parameters
    random.seed(4711)

    ### parse command line args
    parser = argparse.ArgumentParser(description="training model for udacity carnd project 3")
    parser.add_argument('--epochs', type=int, default=4, help='# of training epochs')
    parser.add_argument('--limit_samples', type=int, default=-1, help='# of samples to')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--plot_train_samples', type=int, default=-1,
                        help='plot given number of first training samples in first batch')
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    limit_samples = args.limit_samples
    plot_train_samples_count = args.plot_train_samples

    ### load data sets
    #drive_log1 = load_drive_log('../drivelog1/driving_log.csv')
    drive_log2 = load_drive_log('../drivelog2/driving_log.csv', header=0)
    #drive_log3 = load_drive_log('../drivelog3/driving_log.csv')
    #drive_log4 = load_drive_log('../drivelog4/driving_log.csv')
    #drive_log_all = pd.concat([drive_log1, drive_log2, drive_log3])
    drive_log_all = drive_log2 #pd.concat([drive_log1, drive_log2, drive_log3])

    ### create model and train it
    steering_model = create_model_nvidia()
    train_model(steering_model, drive_log_all)
