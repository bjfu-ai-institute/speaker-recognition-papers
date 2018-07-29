import pyasv.speech_processing
from pyasv import Config
from pyasv.model import ctdnn
from pyasv.data_manage import DataManage

root = "/opt/user1/fhq/asr/data-url/c863/"


config = Config(config_path="./test.json")

train_frames, train_labels = pyasv.speech_processing.ext_fbank_feature(root + 'train')
enroll_frames, enroll_labels = pyasv.speech_processing.ext_fbank_feature(root + 'enroll')

train = DataManage(train_frames, train_labels, config)
validation = DataManage(enroll_frames, enroll_labels, config)

ctdnn.run(config, train, validation)
