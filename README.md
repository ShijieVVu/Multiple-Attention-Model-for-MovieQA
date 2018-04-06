# Multiple-Attention-Model-for-MovieQA

## Extracting Image Features from Movie Clips
1. Download the MovieQA dataset
2. Install ffmpeg, keras, tensorflow
3. ```git clone https://github.com/fchollet/deep-learning-models```
4. Edit paths in movie2feature.py
5. Run```Python movie2feature.py```
When I use the setting where videos are sliced every 2 seconds, the final processed data is 73GB. and the overall process took 5 hours to run on a GTX 1080Ti. 
