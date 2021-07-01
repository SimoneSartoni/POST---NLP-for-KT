import subprocess
def git(*args):
    return subprocess.check_call(['git'] + list(args))


def pip(*args):
    return subprocess.check_call(['pip'] + list(args))


def python(*args):
    return subprocess.check_call(['python'] + list(args))


def cp(*args):
    return subprocess.check_call(['cp'] + list(args))