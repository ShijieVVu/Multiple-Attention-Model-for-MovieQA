# Process video as image sequences

# Path to your downloaded MovieQA directory
MOVIEQA_PATH = ""
# Path to your deep model directory "https://github.com/fchollet/deep-learning-models"
DEEPMODEL_PATH = ""
# Path to your video feature output directory
OUTPUT_PATH = ""

import sys

sys.path.insert(0, r'C:\Users\WUSHI\github\MovieQA_benchmark')
sys.path.insert(0, r'C:\Users\WUSHI\github\deep-learning-models')
import data_loader

from os import listdir, curdir, unlink, rmdir
from os.path import join, isdir
from pickle import dump

import numpy as np
from keras.models import Model
from keras.preprocessing import image

from imagenet_utils import preprocess_input
from vgg19 import VGG19
from time import time

from os import chdir, makedirs
from os.path import join, isdir, exists
from subprocess import call

frames_per_sec = 2
video_base = r"D:\video_clips"
output_dir = r"D:\data_processed"
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

mqa = data_loader.DataLoader()

chdir(output_dir)
vl_qa, qa = mqa.get_video_list('train', 'qa_clips')

print("Total number of movie QAs are {}".format(len(qa)))

videos = set([])
for video_clips in vl_qa.values():
    for video in video_clips:
        videos.add(video)

# Get video clips in QA
videos = list(videos)
print("There are total of {} videos".format(len(videos)))

for i, video_name in enumerate(videos, 0):
    if i % 10 == 0:
        print("Finished {}th of {} movie conversion".format(i, len(videos)))
    tar_name = video_name[:video_name.find('.')]
    # abs_tar_path = join(video_base, tar_name)
    # if not isdir(abs_tar_path):
    #     call(['tar', '-xf', '{}.tar'.format(abs_tar_path), '-C', video_base])
    input_path = join(video_base, tar_name, video_name)
    feature_dir = join(output_dir, video_name)

    # start = time()
    chdir(output_dir)
    if not exists("{}features.p".format(video_name)):
        input_seq = []
        makedirs(feature_dir)
        chdir(feature_dir)
        # Generate images from movie clips
        call(['ffmpeg', '-loglevel', 'panic', '-i', '{}'.format(input_path), '-r', '{}'.format(frames_per_sec),
              '{}%04d.png'.format(video_name)])

        # Convert images to features
        features = []
        for file in listdir(curdir):
            if file.endswith("png"):
                img = image.load_img(file, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                img_features = model.predict(x)[0, ...]
                features.append(img_features.reshape(-1, 512))
                unlink(file)
        chdir(output_dir)
        dump(np.array(features), open("{}features.p".format(video_name), "wb"))
        rmdir(feature_dir)
    # print(time() - start)
