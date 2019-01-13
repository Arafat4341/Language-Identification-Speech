'''
    This class is used in pipeline.py in order to properly call other
    functions that use ArgumentParses as their parameter. To avoid
    changing the code just for the pipeline.py, it is only necessary
    to extend the dictionary to be able to call the value by
    specifying the key not only by dict["key"], but also by dict.key
    since that how ArgumentParses does it.
'''

# https://stackoverflow.com/a/32107024/9344265
class ExtendedDict(dict):
    """
    Example:
    m = ExtendedDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(ExtendedDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(ExtendedDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ExtendedDict, self).__delitem__(key)
        del self.__dict__[key]