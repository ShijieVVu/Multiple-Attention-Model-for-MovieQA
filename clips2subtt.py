import sys
from enum import Enum
from pickle import dump


class LineType(Enum):
    TIME = 1
    SUB = 2
    OTHER = 3


# Path to your downloaded MovieQA directory
MOVIEQA_PATH = "/Users/shijiewu/github/MovieQA_benchmark"
# Path to your video feature output directory
OUTPUT_PATH = "/Users/shijiewu/github/MovieQA_benchmark/subtt"

sys.path.insert(0, MOVIEQA_PATH)

import data_loader

mqa = data_loader.DataLoader()
vl_qa, qa_pairs = mqa.get_video_list('train', 'qa_clips')


def get_line_type(line):
    if ' --> ' in line:
        return LineType.TIME
    elif line.rstrip().isdigit():
        return LineType.OTHER
    elif line.rstrip() == '':
        return LineType.OTHER
    else:
        return LineType.SUB


def string2second(time_string):
    (h, m, s) = time_string[:time_string.find(',')].split(':')
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    return seconds


print("There are total of {} scripts.".format(len(qa_pairs)))

for i, qa in enumerate(qa_pairs[:3], 0):
    if i % 10 == 0:
        print("{}th subtitle processed".format(i))
    imdb_key = qa.imdb_key
    video_name = qa[5][0]
    start_frame = video_name[video_name.find('-') + 1:][:video_name[video_name.find('-') + 1:].find('.')]
    end_frame = video_name[video_name.find('-') + 1:][video_name[video_name.find('-') + 1:].find('-') + 1:][
                :video_name[video_name.find('-') + 1:][video_name[video_name.find('-') + 1:].find('-') + 1:].find('.')]

    shot_bound = MOVIEQA_PATH + '/story/matidx/{}.matidx'.format(imdb_key)
    with open(shot_bound, 'r+') as f:
        time_cuts = f.readlines()

    start_flag = end_flag = False
    start_time = None
    end_time = None
    for cuts in time_cuts:
        frame, second = cuts.split(' ')
        if frame == start_frame:
            start_flag = True
            start_time = int(second[:second.find('.')])

        if frame == end_frame:
            end_flag = True
            end_time = int(second[:second.find('.')])

        if start_flag and end_flag:
            break

    subtt = MOVIEQA_PATH + '/' + mqa.movies_map[imdb_key].text[2]
    with open(subtt, 'r+') as f:
        content = f.readlines()

    recording = False
    subtitles = []
    for line in content:
        line_type = get_line_type(line)
        if recording is False:
            if line_type is LineType.TIME and string2second(line) > start_time:
                recording = True
        else:
            if line_type is LineType.SUB:
                subtitles.append(line)
            elif line_type is LineType.TIME and string2second(line) > end_time:
                break
    dump(subtitles, open(OUTPUT_PATH + '/{}.p'.format(video_name), 'wb'))
