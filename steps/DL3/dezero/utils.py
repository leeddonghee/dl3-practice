import os
import subprocess
from dezero.core_simple import Variable


def _dot_var(v, verbose=False):
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name:
            name += ': '
        name += str(v.data.shape) + ' ' + str(v.dtype)
    return f'"{id(v)}" [label="{name}", color=orange, style=filled]\n'


def _dot_func(f):
    txt = f'"{id(f)}" [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        txt += f'"{id(x)}" -> "{id(f)}"\n'
    for y in f.outputs:
        txt += f'"{id(f)}" -> "{id(y())}"\n'
    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, 'graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    output_path = os.path.join(os.path.dirname(__file__), to_file)
    cmd = f'dot {graph_path} -Tpng -o {output_path}'
    subprocess.run(cmd, shell=True)
