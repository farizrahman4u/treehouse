import ast

_cache = {}


def get_output(code, input=None, context=[]):
    if type(context) is list:
        context = {x.__name__ : x for x in context}
    output = '__no__output__'
    if code in _cache:
        compiled = _cache[code]
    else:
        block = ast.parse(code, mode='exec')
        compiled = compile(block, '<string>', mode='exec')
        _cache[code] = compiled
    context['input'] = input
    context['output'] = output
    globals().update(context)
    exec(compiled, globals())
    return globals()['output']
