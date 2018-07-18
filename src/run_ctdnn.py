import sys
sys.path.append("..")
from src import Config
from src.model import CTDnn
from src.speech_processing import ext_fbank_feature


train_frames, train_labels = ext_fbank_feature('/opt/user1/fhq/asr/data-url/c863/train')
enroll_frames, enroll_labels = ext_fbank_feature('/opt/user1/fhq/asr/data-url/c863/enroll')
test_frames, test_labels = ext_fbank_feature('/opt/user1/fhq/asr/data-url/c863/test')

config = Config(
    name='CTDnn-config',
    batch_size=65,
    n_gpu=4,
    max_step=100,
    n_speaker=166,
    is_big_dataset=False,
    url_of_bigdataset_temp_file=None,
    learning_rate=1e-3,
    save_path='/opt/user1/fhq/save/ctdnn')
config.save('/opt/user1/fhq/save/config-ctdnn', name='ctdnn-config')

model = CTDnn(config)
model.run(train_frames=train_frames, 
          train_labels=train_labels,
          enroll_frames=enroll_frames,
          enroll_labels=enroll_labels,
          test_frames=test_frames,
          test_labels=test_labels,
          need_prediction_now=False)
