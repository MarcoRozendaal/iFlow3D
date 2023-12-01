"""
In this file, we defined a custom object that is used when creating nested dictionary with
arbitrary depths.
Then it can return a normal dictionary when the creation is finished
"""

from collections import defaultdict


class NestedDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))

    def get_dict(self):
        def default_to_regular(d):
            if isinstance(d, defaultdict):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d
        return default_to_regular(self)
