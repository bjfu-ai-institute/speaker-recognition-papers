class Model(object):
    def __init__(self, config):
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
