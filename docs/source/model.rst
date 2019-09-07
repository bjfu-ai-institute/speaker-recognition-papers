Model
=====

.. note::
    Basic setting in config (every model needs)
    
    * model name
    * speaker number
    * max training step
    * number of gpus. if equal to zero, we'll run model without gpu.
    * save path
    * learning rate
    * batch size

CTDNN
-----

`FULL-INFO TRAINING FOR DEEP SPEAKER FEATURE LEARNING
<https://github.com/vzxxbacq/speaker-recognition-papers/blob/master/papers/Full_info_deep_speaker_feature_learning.pdf>`_.
*Lantian Li, Zhiyuan Tang, Dong Wang, Thomas Fang Zheng*

.. autoclass:: pyasv.model.ctdnn.CTDnn
    :members:

    .. automethod:: __init__

.. autofunction:: pyasv.model.ctdnn.run

DeepSpeaker
-----------

`Deep Speaker: an End-to-End Neural Speaker Embedding System
<https://github.com/vzxxbacq/speaker-recognition-papers/blob/master/papers/Deep%20Speaker%20an%20End-to-End%20Neural%20Speaker%20Embedding%20System.pdf>`_.
*Chao Li∗, Xiaokong Ma∗, Bing Jiang∗, Xiangang Li ∗ Xuewei Zhang, Xiao Liu, Ying Cao, Ajay Kannan, Zhenyao Zhu*

.. note::
    Extra setting in config:
    
    * weight decay for conv layer
    * weight decay for fc layer
    * epsilon for bn layer

.. autoclass:: pyasv.model.deep_speaker.DeepSpeaker
    :members:

    .. automethod:: __init__

.. autofunction:: pyasv.model.deep_speaker.run

Max Feature Map model
---------------------

`DEEP CNN BASED FEATURE EXTRACTOR FOR TEXT-PROMPTED SPEAKER RECOGNITION
<https://github.com/vzxxbacq/speaker-recognition-papers/blob/master/papers/DEEP_CNN_BASED_FEATURE_EXTRACTOR_FOR_TEXT-PROMPTED_SPEAKER.pdf>`_.
*Sergey Novoselov1,2, Oleg Kudashev2, Vadim Shchemelinin1, Ivan Kremnev3, Galina Lavrentyeva1*

.. autoclass:: pyasv.model.max_feature_map_dnn_model.MaxFeatureMapDnn
    :members:

    .. automethod:: __init__

.. autofunction:: pyasv.model.max_feature_map_dnn_model.run