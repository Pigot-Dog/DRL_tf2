import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile


def is_return_code_zero(args):
    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_under_git_control():
    return is_return_code_zero(['git', 'rev-parse'])


def prepare_output_dir(args, user_specified_dir=None, argv=None, time_format='%Y%m%dT%H%M%S.%f', suffix=""):
    if suffix is not "":
        suffix = "_" + suffix
    time_str = datetime.datetime.now().strftime(time_format) + suffix
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeWarning(
                    '{} is not a directory'.format(user_specified_dir)
                )
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise  RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)

    # Save all the arguments 保存args参数
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args, sort_keys=True, indent=4, separators=(',', ':')))

    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ), sort_keys=True, indent=4, separators=(',', ':')))

    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
            f.write(subprocess.check_output('git rev-parse HEAD'.split()))

        # Save `git status`
        with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
            f.write(subprocess.check_output('git status'.split()))

        # Save `git log`
        with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
            f.write(subprocess.check_output('git log'.split()))

        # Save `git diff`
        with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
            f.write(subprocess.check_output('git diff'.split()))

    return outdir



