## Introduction

These are the slightly modified tensorflow/python implementation of recent speaker recognition papers. Please tell me if it is copyright infringement, I'll delete this repo as soon as I can. Thx.

The file structure is as follows：
```
---FolderName (folder)

------models (folder)
---------DataManage.py (Class of batch managing.)
---------model.py (Class of this model)

------PaperName.pdf 
------config.py (Settings. e.g. save path, learning rate)
```
If you want run these code on your computer, you only need to modify config.py and write code as follow:

```python
import FolderName.models as model

M = model.Model()
M.run(train_data, train_label, 
      enroll_data=None, enroll_label=None, test_data=None, test_label=None)

# If the enroll_data or test_data is none, you will only get the trained model. 
# You can do this when you needn't the ACC/EER information.
```

## TODO

* Implement papers of ICASSP 2018 & Interspeech 2018.
* Compare each model on a same dataset.

## Implemented papers:

* L. Li, Z. Tang, D. Wang, T. Zheng, "Deep Speaker Feature Learning for Text-Independent Speaker Verification." 
* L. Li, Z. Tang, D. Wang, T. Zheng, "Full-info Training for Deep Speaker Feature Learning," ICASSP 2018.
* C. Li, X. Ma, B. Jiang, X. Li, X. Zhang, X. Liu, Y. Cao, A. Kannan, Z. Zhu, "Deep Speaker: an End-to-End Neural Speaker Embedding System."
* Sergey Novoselov, Oleg Kudashev, Vadim Shchemelinin, Ivan Kremnev, Galina Lavrentyeva, "DEEP CNN BASED FEATURE EXTRACTOR FOR TEXT-PROMPTED SPEAKERRECOGNITION."