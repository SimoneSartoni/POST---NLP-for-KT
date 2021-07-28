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

