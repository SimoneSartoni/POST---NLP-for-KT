import ast
import os
from datetime import datetime
import numpy as np
import subprocess


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def pip(*args):
    return subprocess.check_call(['pip'] + list(args))


def python(*args):
    return subprocess.check_call(['python'] + list(args))


def cp(*args):
    return subprocess.check_call(['cp'] + list(args))


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            if dd is not None:
                if isinstance(dd, str):
                    f.write(str(dd.encode('utf8'))+'\n')
                else:
                    f.write(str(dd)+'\n')


def try_parsing_date(text):
    try:
        return datetime.fromtimestamp(int(text)//1000).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass
    except TypeError:
        pass
    try:
        x = datetime.fromtimestamp(str(text))
        print("here:" + x)
        return x
    except ValueError:
        pass
    except TypeError:
        pass
    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
        try:
            return datetime.strptime(str(text), fmt).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
    raise ValueError('no valid date format found')


def parse_datetime_list(astr, debug=False):
    try:
        tree = ast.parse(astr)
    except SyntaxError:
        raise ValueError(astr)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.Expr, ast.Dict, ast.Str, ast.Attribute, ast.Num, ast.Name, ast.Load,
                             ast.Tuple)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'datetime':
            continue
        if debug:
            attrs = [attr for attr in dir(node) if not attr.startswith('__')]
            print(node)
            for attrname in attrs:
                print('    {k} ==> {v}'.format(k=attrname, v=getattr(node, attrname)))
        raise ValueError(astr)
    return eval(astr)


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
