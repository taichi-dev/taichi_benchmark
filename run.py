# -*- coding: utf-8 -*-

# -- stdlib --
import argparse
import logging
import dataclasses, json


# -- third party --
# -- own --
from utils import logconfig
import core.runner


# -- code --
log = logging.getLogger(__name__)


class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)


def save_json(results, path):
    import json
    with open(path, 'w') as f:
        json.dump(results, f, cls=EnhancedJSONEncoder)


def main():
    parser = argparse.ArgumentParser('taichi-benchmark')
    parser.add_argument('--log', default='INFO')
    parser.add_argument('--upload-auth', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    options = parser.parse_args()
    logconfig.init(getattr(logging, options.log))

    results = core.runner.run_benchmarks()
    if options.save:
        log.info('Saving results to %s', options.save)
        save_json(results, options.save)

if __name__ == '__main__':
    main()
