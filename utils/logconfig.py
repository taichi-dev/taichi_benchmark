# -*- coding: utf-8 -*-

# -- stdlib --
import logging
import sys

# -- third party --
from .escapes import escape_codes


# -- code --
class SimpleLogFormatter(logging.Formatter):
    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color
        self.color_mapping = {
            'CRITICAL': 'bold_red',
            'ERROR': 'red',
            'WARNING': 'yellow',
            'INFO': 'green',
            'DEBUG': 'blue',
        }

    def format(self, rec):

        if rec.exc_info:
            s = []
            s.append('>>>>>>' + '-' * 74)
            s.append(self._format(rec))
            import traceback
            s.append(''.join(traceback.format_exception(*rec.exc_info)).strip())
            s.append('<<<<<<' + '-' * 74)
            return '\n'.join(s)
        else:
            return self._format(rec)

    def _format(self, rec):
        import time
        rec.message = rec.getMessage()
        lvl = rec.levelname
        prefix = '[{} {} {}:{}]'.format(
            lvl[0],
            time.strftime('%y%m%d %H:%M:%S'),
            rec.module,
            rec.lineno,
        )
        if self.use_color:
            E = escape_codes
            M = self.color_mapping
            prefix = f"{E[M[lvl]]}{prefix}{E['reset']}"

        return f'{prefix} {rec.message}'


def init(level):
    root = logging.getLogger()
    root.setLevel(level)

    fmter = SimpleLogFormatter()
    std = logging.StreamHandler(stream=sys.stdout)
    std.setFormatter(fmter)
    root.addHandler(std)

    logging.getLogger('sentry.errors').setLevel(1000)
