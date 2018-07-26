Config
======

The config class is used for set our model and operation.
In general, we only need a part of his parameters. So, we
just need to set the needed parameters when we create a config class.

::

    from pyasv import Config


    # The normal way to create a config.
    config = Config(name='CTDnn-config',
                    batch_size=65,
                    n_gpu=4,
                    max_step=100,
                    n_speaker=166,
                    is_big_dataset=False,
                    url_of_bigdataset_temp_file=None,
                             learning_rate=1e-3,
                             save_path='/opt/user1/fhq/save/ctdnn')
    # Load saved config.
    config.save(name='ctdnn-config')
    # Use config path to create a config
    config = Config(config_path='/opt/user1/fhq/save/ctdnn/ctdnn-config.pkl')



.. autoclass:: pyasv.Config
    :members:

    .. automethod:: __init__