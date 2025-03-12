from .mellow import Mellow

def get_model_class(model_type):
    if model_type == 'Mellow':
        return Mellow
    else:
        raise NotImplementedError
