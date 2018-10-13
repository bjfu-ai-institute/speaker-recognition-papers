class Model(object):
    def __init__(self, config):
        """Create deep speaker model.

        Parameters
        ----------
        config : ``config`` class.
            The config of ctdnn model, no extra need.
        out_channel : ``list``
            The out channel of each res_block.::

                out_channel = [64, 128, 256, 512]

        Notes
        -----
        There is no predict operation in deep speaker model.
        Because we use the output of last layer as speaker vector.
        we can use DeepSpeaker.feature to get these vector.
        """
        self.config = config

    def __call__(self, x, y, training=True):
        if training:
            self._build_train_graph()
        else:
            self._build_test_graph()
        pass

    def inference(self, x):
        pass

    def _build_train_graph(self):
        pass

    def _build_test_graph(self):
        pass
