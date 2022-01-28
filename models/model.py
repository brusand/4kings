#from api.rest import Rest


class AbstractModel():
    resource_name = ''

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)