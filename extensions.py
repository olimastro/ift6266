import cPickle as pkl
from blocks.extensions import SimpleExtension


class SaveParams(SimpleExtension) :
    def __init__(self, name, **kwargs) :
        super(SaveParams, self).__init__(**kwargs)
        self.name = name

    def do(self, which_callback, *args) :
        params = self.main_loop.model.get_parameter_values()

        f = open(self.name+"_params.pkl", 'w')
        pkl.dump(params, f)
        f.close()
