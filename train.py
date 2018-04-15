from keras.models import load_model
from data_generator import DataGenerator
import sys
sys.path.insert(0, r'C:\Users\WUSHI\github\MovieQA_benchmark')
import data_loader
model = load_model(r"./model/dan.h5")

mqa = data_loader.DataLoader()
vl_qa, training_qas = mqa.get_video_list('train', 'qa_clips')
training_qa = training_qas[:32]
training_generator = DataGenerator(training_qa, batch_size=16)
validation_qa = training_qas[4096:4224]
validation_generator = DataGenerator(validation_qa, batch_size=16)
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10,
                    use_multiprocessing=False, workers=1)
