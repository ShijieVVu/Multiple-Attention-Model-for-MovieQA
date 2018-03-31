# Convert mp4 into sequences of images
# Place this in your MovieQA directory
# Requirements: ffmpeg

from os import chdir, makedirs
from os.path import join, isdir, exists
from subprocess import call

import data_loader

mqa = data_loader.DataLoader()

# Your video_clips directory
video_base = ""
# Your data_processed directory
output_dir = ""
# Frames per second
frames_per_sec = 1

chdir(output_dir)
vl_qa, qa = mqa.get_video_list('train', 'qa_clips')

print("Total number of movie conversion is {}".format(len(qa)))
for i, (qIndex, qa_videos) in enumerate(vl_qa.items()):
    if i % 10 == 0:
        print("Finished {}th movie conversion".format(i))
    for video_name in qa_videos:
        tar_name = video_name[:video_name.find('.')]
        abs_tar_path = join(video_base, tar_name)
        if not isdir(abs_tar_path):
            call(['tar', '-xf', '{}.tar'.format(abs_tar_path), '-C', video_base])
        input_path = join(video_base, tar_name, video_name)
        qIndex = qIndex.replace(':', '_')
        output_qa_dir = join(output_dir, qIndex)
        if not exists(output_qa_dir):
            makedirs(output_qa_dir)
            chdir(output_qa_dir)
            call(['ffmpeg', '-loglevel', 'panic', '-i', '{}'.format(input_path), '-r', '{}'.format(frames_per_sec),
                  '{}%04d.png'.format(qIndex.replace(':', '_'))])
            chdir(output_dir)
