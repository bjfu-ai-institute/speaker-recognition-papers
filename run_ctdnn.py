import pyasv.speech_processing
from pyasv.model import CTDnn
from pyasv import Config


root = "/opt/user1/fhq/asr/data-url/c863/"


config = Config(config_path="./test.json")

train_frames, train_labels = pyasv.speech_processing.ext_fbank_feature(root + 'train')
enroll_frames, enroll_labels = pyasv.speech_processing.ext_fbank_feature(root + 'enroll')
test_frames, test_labels = pyasv.speech_processing.ext_fbank_feature(root + 'test')

m = CTDnn(config)
m.run(train_frames, train_labels, enroll_frames, enroll_labels, test_frames, test_labels)
