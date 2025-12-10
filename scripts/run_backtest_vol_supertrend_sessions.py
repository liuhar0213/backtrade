import os
import runpy
import logging


def main():
    logging.basicConfig(level=logging.INFO)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    target = os.path.join(repo_root, 'backtest_vol_supertrend_sessions.py')
    logging.info("Executing legacy backtest: %s", target)
    runpy.run_path(target, run_name="__main__")
    return True


if __name__ == '__main__':
    main()
