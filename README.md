# Speaker recognition papers' implementation
These are the slightly modified tensorflow/python implementation of recent speaker recognition papers.

These codes do not include feature extraction and main program. If you want to use some models, you need to write the extraction and main program by yourself.

The file structure is as follows：
```
---PaperName (folder)

------models (folder)
---------DataManage.py (Class of batch managing.)
---------model.py (Class of this model)

------config.py (Settings. e.g. save path, learning rate)
```
If you want run these codes on your computer, you only need to modify config.oy and write code as follow:
```python
import PaperName.models as model

M = models.model()
M.run(train_data, train_label, test_data=None, test_label=None)

# If the test_data or test_label is none, you will only get the trained model. 
# You can do this when you needn't the ACC/EER information.
```

## [Deep Speaker Feature Learning for Text-Independent Speaker Verification](https://github.com/vzxxbacq/speaker-recognition-papers/blob/master/CT-DNN/Deep_Speaker_Feature_Learning_for_Text-Independent.pdf) & [Full-info Training for Deep Speaker Feature Learning](https://github.com/vzxxbacq/speaker-recognition-papers/blob/master/CT-DNN/Full_info_deep_speaker_feature_learning.pdf)

- Author:
```
    Lantian Li,
    Zhiyuan Tang,  
    Dong Wang, 
    Thomas Fang Zheng  
```
- Organization:
```
    Center for Speech and Language Technologies.
    Research Institute of Information Technology.
    Department of Computer Science and Technology.
    Tsinghua University.
```
## [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://github.com/vzxxbacq/speaker-recognition-papers/blob/master/DeepSpeaker/Deep%20Speaker%20an%20End-to-End%20Neural%20Speaker%20Embedding%20System.pdf)

- Author:
```
    Chao Li, Xiaokong Ma,
    Bing Jiang, Xiangang Li,
    Xuewei Zhang, Xiao Liu,
    Ying Cao, Ajay Kannan, Zhenyao Zhu
```
- Organization:
```
    Baidu Inc.
```
