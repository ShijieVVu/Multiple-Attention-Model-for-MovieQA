from pickle import load

video_path = "/Users/shijiewu/github/MovieQA_benchmark/data_processed"
# Input video
# Input: data_path of pickled image files
# Output: Vn -- video feature of shape (C * 49, 512), C is length of image sequence
def video_input(video_path):
    video_path

# Input text
# Input: path of text files
# Output: Ut -- text feature shape (T, 256), T is length of hidden units

# Video attention
# Input: Vn --  video feature of shape (C * 49, 512), C is length of image sequence
#        m  --  memory vector encoding information attended, with shape of (T, 256)
# Output: v --  visual context vector, with shape (T, 256)

# Textual attention
# Input: Ut -- text feature shape (T, 256), T is length of hidden units
#        m  --  memory vector encoding information attended, with shape of (T, 256)
# Output: u -- textual context vector, with shape (T, 256)

# r-DAN
# Input: u -- textual context vector, with shape (T, 256)
#        v --  visual context vector, with shape (T, 256)
#        m --  old memory vector encoding information attended, with shape of (T, 256)
# Output: m --  new memory vector encoding information attended, with shape of (T, 256)

# prediction
# Input: m -- memory vector encoding information attended, with shape of (T, 256)
# Output: scalar -- probability of it being the right answer

