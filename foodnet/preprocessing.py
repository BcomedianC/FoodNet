import collections
import multiprocessing as mp
import os
import shutil
import stat
from collections import defaultdict
from os import listdir
from os.path import join

# import tools.image_gen_extended as T
import keras_preprocessing.image.image_data_generator as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical

from foodnet.datasets import Datasets


class Preprocessing(object):
    """
    This class manages the construction of the dataset. Any data cleaning, pipelining,
    and reading from disk is performed here.
    """

    def __init__(self):
        self.DATA_DIR = '../data/food-101/images/'
        self.num_processes = 6
        self.pool = mp.Pool(processes=self.num_processes)

    """
    Loads the data into disk using the Datasets utility class.
    """

    def load_data(self):
        datasets = Datasets('http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz')
        datasets.download("food101.tgz")

    """
    Given the directory corresponding to a class of images, this function
    returns a formatted version of the name.
    """

    def __parseClassName(self, dir):
        tokens = dir.split("_")
        class_name = ""
        for token in tokens:
            class_name += token.capitalize() + " "
        return class_name[:-1]

    """
    Displays a random image from each class to provide a better understanding of the data.
    """

    def peek_dataset(self):
        ROWS = 17
        COLS = 6
        FIG_WIDTH = 15
        FIG_HEIGHT = 25
        fig, ax = plt.subplots(ROWS, COLS, figsize=(FIG_HEIGHT, FIG_WIDTH))
        fig.suptitle("A random image from each food class.")

        FOOD_IMAGE_DIRS = sorted(os.listdir(self.DATA_DIR))

        for i in range(ROWS):
            for j in range(COLS):
                try:
                    curr_image_dir = FOOD_IMAGE_DIRS[i * COLS + j]
                except:
                    break
                images = os.listdir(os.path.join(self.DATA_DIR, curr_image_dir))
                selected_image = np.random.choice(images)
                image = plt.imread(os.path.join(self.DATA_DIR, curr_image_dir, selected_image))
                ax[i][j].imshow(image)
                ec = (0, .6, .1)
                fc = (0, .7, .2)
                ax[i][j].text(0, -20, self.__parseClassName(curr_image_dir), size=10, rotation=0, ha="right", va="top",
                              bbox=dict(boxstyle="round", ec=ec, fc=fc))

        plt.setp(ax, xticks=[], yticks=[])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    """
    Returns a tuple containing two maps: from the class to the index and vice-versa
    """

    def create_classes_map(self):
        classes_to_indices = {}
        indices_to_classes = {}

        with(open('../data/food-101/meta/classes.txt', 'r')) as classes_txt:
            classes = [cl.strip() for cl in classes_txt.readlines()]
            classes_to_indices = dict(zip(classes, range(len(classes))))
            indices_to_classes = dict((v, k) for k, v in classes_to_indices.items())

        print(collections.OrderedDict(sorted(classes_to_indices.items())))
        print(collections.OrderedDict(sorted(indices_to_classes.items())))

        return classes_to_indices, indices_to_classes

    """
    Creates a train and test set directory with the split provided by the metadata.
    """

    def create_train_test_split(self):
        if not os.path.isdir('../data/food-101/test') and not os.path.isdir('../data/food-101/train'):
            def copytree(src, dst, symlinks=False, ignore=None):
                if not os.path.exists(dst):
                    os.makedirs(dst)
                    shutil.copystat(src, dst)
                lst = os.listdir(src)
                if ignore:
                    excl = ignore(src, lst)
                    lst = [x for x in lst if x not in excl]
                for item in lst:
                    s = os.path.join(src, item)
                    d = os.path.join(dst, item)
                    if symlinks and os.path.islink(s):
                        if os.path.lexists(d):
                            os.remove(d)
                        os.symlink(os.readlink(s), d)
                        try:
                            st = os.lstat(s)
                            mode = stat.S_IMODE(st.st_mode)
                            os.lchmod(d, mode)
                        except:
                            pass  # lchmod not available
                    elif os.path.isdir(s):
                        copytree(s, d, symlinks, ignore)
                    else:
                        shutil.copy2(s, d)

            def generate_dir_file_map(path):
                dir_files = defaultdict(list)
                with open(path, 'r') as txt:
                    files = [l.strip() for l in txt.readlines()]
                    for f in files:
                        dir_name, id = f.split('/')
                        dir_files[dir_name].append(id + '.jpg')
                return dir_files

            train_dir_files = generate_dir_file_map('../data/food-101/meta/train.txt')
            test_dir_files = generate_dir_file_map('../data/food-101/meta/test.txt')

            def ignore_train(d, filenames):
                print(d)
                subdir = d.split('/')[-1]
                to_ignore = train_dir_files[subdir]
                return to_ignore

            def ignore_test(d, filenames):
                print(d)
                subdir = d.split('/')[-1]
                to_ignore = test_dir_files[subdir]
                return to_ignore

            copytree('../data/food-101/images', '../data/food-101/test', ignore=ignore_train)
            copytree('../data/food-101/images', '../data/food-101/train', ignore=ignore_test)
        else:
            print('Train/Test directories already exist!')

    def resize_and_load_images(self, image_dir, min_side):
        all_images = []
        all_classes = []
        resize_count = 0
        invalid_count = 0
        classes_to_indices, indices_to_classes = self.create_classes_map()
        for i, sub_dir in enumerate(listdir(image_dir)):
            images = listdir(join(image_dir, sub_dir))
            cl_ind = classes_to_indices[sub_dir]
            for image in images:
                # image_arr = img.imread(join(image_dir, sub_dir, image))
                image_arr = Image.open(join(image_dir, sub_dir, image))
                # image_arr = np.array(image_arr.getdata())
                resized_image_arr = image_arr
                try:
                    w, h = image_arr.size
                    if w < min_side:
                        wpercent = (min_side / float(w))
                        hsize = int((float(h) * float(wpercent)))
                        # print('new dims:', min_side, hsize)
                        # resized_image_arr = imresize(image_arr, (min_side, hsize))
                        resized_image_arr = resized_image_arr.resize((min_side, hsize))
                        resized_image_arr = np.array(resized_image_arr.getdata())
                        resize_count += 1
                    elif h < min_side:
                        hpercent = (min_side / float(h))
                        wsize = int((float(w) * float(hpercent)))
                        # print('new dims:', wsize, min_side)
                        # resized_image_arr = imresize(image_arr, (wsize, min_side))
                        resized_image_arr = resized_image_arr.resize((wsize, min_side))
                        resized_image_arr = np.array(resized_image_arr.getdata())
                        resize_count += 1

                    image_arr.close()
                    all_images.append(resized_image_arr)
                    all_classes.append(cl_ind)
                except:
                    print('Unable to read image: ', sub_dir, image)
                    invalid_count += 1
                    image_arr.close()

        print(len(all_images), 'images loaded')
        print(resize_count, 'images resized')
        print(invalid_count, 'images skipped')
        return np.array(all_images), np.array(all_classes)

    def one_hot_encode_labels(self, train_labels, test_labels):
        n_classes = 101
        train_labels_cat = to_categorical(train_labels, nb_classes=n_classes)
        test_labels_cat = to_categorical(test_labels, nb_classes=n_classes)
        return train_labels_cat, test_labels_cat

    def augment_data(self, X_train, X_test, y_train_cat, y_test_cat):
        augmented_train_data = T.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            zoom_range=[.8, 1],
            channel_shift_range=30,
            fill_mode='reflect'
        )

        augmented_train_data.config['random_crop_size'] = (299, 299)
        augmented_train_data.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
        augmented_train_gen = augmented_train_data.flow(X_train, y_train_cat, batch_size=64, seed=11, pool=self.pool)

        augmented_test_data = T.ImageDataGenerator()
        augmented_test_data.config['random_crop_size'] = (299, 299)
        augmented_test_data.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
        augmented_test_gen = augmented_test_data.flow(X_test, y_test_cat, batch_size=64, seed=11, pool=self.pool)

        return augmented_train_gen, augmented_test_gen


preprocessing = Preprocessing()
X_train, y_train = preprocessing.resize_and_load_images('../data/food-101/train', 299)
X_test, y_test = preprocessing.resize_and_load_images('../data/food-101/test', 299)

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)
