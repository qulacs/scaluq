import doctest
import importlib
import sys

space = sys.argv[1]
prec = sys.argv[2]

import scaluq
scaluq_sub = importlib.import_module(f'scaluq.{space}.{prec}')

globs = globals().copy()
def add_to_globs(module):
    for name in dir(module):
        if not name.startswith('_'):
            globs[name] = getattr(module, name)
add_to_globs(scaluq_sub)
add_to_globs(getattr(scaluq_sub, 'gate'))


def test_object(obj, name):
    if obj.__doc__ and '>>> ' in obj.__doc__:
        print(f'Running doctests for {name}...')
        doctest.run_docstring_examples(obj, globs, name=name, optionflags=doctest.NORMALIZE_WHITESPACE)
    if type(obj) in [int, float, str, bool, type(None)]:
        return
    for attr in dir(obj):
        if not attr.startswith('_'):
            test_object(getattr(obj, attr), f'{name}.{attr}')

test_object(scaluq_sub, f'scaluq.{space}.{prec}')
