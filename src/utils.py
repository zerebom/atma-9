import os
from contextlib import contextmanager
from time import time
from pathlib import Path


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


def savefig(fig, to, OUTPUT_DIR='../output/fig'):
    to = os.path.join(OUTPUT_DIR, to + '.png')
    Path(to).parent.mkdir(exist_ok=True,parents=True)
    print('save to {}'.format(to))
    fig.tight_layout()
    fig.savefig(to, dpi=120)
