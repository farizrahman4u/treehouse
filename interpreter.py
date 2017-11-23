import ast

_cache = {}


def _str(x):
    if hasattr(x, '__name__'):
        return x.__name__
    if type(x) is str:
        return '\'' + x + '\''
    return str(x)


def get_output(code, input=None, context=[]):
    if type(context) is list:
        context = {x.__name__ : x for x in context}
    for k in context:
        exec(k + '=' + 'context[\'' + k + '\']')
    output = '__no__output__'
    if code in _cache:
        compiled = _cache[code]
    else:
        block = ast.parse(code, mode='exec')
        compiled = compile(block, '<string>', mode='exec')
        _cache[code] = compiled
    exec(compiled)
    return output
