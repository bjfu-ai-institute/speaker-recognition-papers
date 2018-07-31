## Introduction

These are the slightly modified tensorflow/python implementation of recent speaker recognition papers.
Please tell me if it is copyright infringement, I'll delete these paper as soon as I can. Our license
only apply to our code these paper is not included. Thx.

The file structure is as follows：
```
|———pyasv
|
|—————model (folder, contain the model)
|
|—————loss (folder, contain the customized loss function)
|
|—————papers (folder, contain the origin paper of most of method)
|
|—————backend(TODO: folder, contain the method of backend)
|
|———data_manage.py (contain some method to manage data)
|
|———speech_processing.py (contain some method to extractfeature and process audio)
|
|———config.py (settings. e.g. save path, learning rate)
```

More info: [Doc](https://vzxxbacq.github.io/speaker-recognition-papers/html/index.html)

If you want run these code on your computer, you only need to write code like this:

```python
from pyasv import Config
from pyasv.speech_processing import ext_mfcc_feature
from pyasv.data_manage import DataManage
from pyasv.model.ctdnn import run

config = pyasv.Config(name='my_ctdnn_model',
                    n_speaker=1e3,
                    batch_size=64,
                    n_gpu=2,
                    max_step=100,
                    is_big_dataset=False,
                    url_of_bigdataset_temp_file=None,
                    learning_rate=1e-3,
                    slide_windows=[4, 4]
                    save_path='/home/my_path')
config.save('./my_config_path')

frames, labels = ext_mfcc_feature('data_set_path', config)
train = DataManage(frames, labels, config)

frames, labels = ext_mfcc_feature('data_set_path', config)
validation = DataManage(frames, labels, config)

run(config, train, validation)
```

## TODO

* Implement papers of ICASSP 2018 & Interspeech 2018.
* Compare each model on a same dataset.

## Implemented papers:

* L. Li, Z. Tang, D. Wang, T. Zheng, "Deep Speaker Feature Learning for Text-Independent Speaker Verification." 
* L. Li, Z. Tang, D. Wang, T. Zheng, "Full-info Training for Deep Speaker Feature Learning," ICASSP 2018.
* C. Li, X. Ma, B. Jiang, X. Li, X. Zhang, X. Liu, Y. Cao, A. Kannan, Z. Zhu, "Deep Speaker: an End-to-End Neural Speaker Embedding System."
* Sergey Novoselov, Oleg Kudashev, Vadim Shchemelinin, Ivan Kremnev, Galina Lavrentyeva, "DEEP CNN BASED FEATURE EXTRACTOR FOR TEXT-PROMPTED SPEAKERRECOGNITION."