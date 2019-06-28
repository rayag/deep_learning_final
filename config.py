
'''
Structure containing metainfo 
'''
class NetworkConfig:
    '''
    name: the name of the model
    logdir: the path to the directory where models will be saved
    model_path: path to the folder where model will be saved and from
                where it will be loaded afterwards
    '''
    def __init__(self, name, logdir, model_path):
        self.name = name
        self.logdir = logdir
        self.model_path = model_path


net1_config = NetworkConfig(name='Network1', logdir='./net1/logdir', model_path='./net1/model')