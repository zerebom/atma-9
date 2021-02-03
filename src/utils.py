import os
from contextlib import contextmanager
from time import time


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix:
        format_str = str(prefix) + format_str
    if suffix:
        format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def savefig(fig, to, OUTPUT_DIR='./tutorial2_outputs'):
    to = os.path.join(OUTPUT_DIR, to + '.png')
    print('save to {}'.format(to))
    fig.tight_layout()
    fig.savefig(to, dpi=120)
