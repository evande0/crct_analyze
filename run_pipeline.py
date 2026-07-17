import argparse
from datetime import datetime

from config import *
from src.utils import *
from src.extract_data import *
from src.process_data import *
from src.analyze_data import *
from src.sensitivity import init_sensitivity, run_sensitivity_analysis


'''
    Run the pipeline by calling 'python3 run_pipeline.py'
'''
def run_full_pipeline(args, logger):
    logger.warning("\n⏳ Begin running data pipeline")
    try:
        use_config = extract_data(force_extract=True)   # Loading from saved data is buggy
        process_data(use_config)
        analyze_data(args.show, args.compact)
        logger.warning("🎉 Data pipeline completed successfully.")
        logger.warning(f"📁 See data and analysis in {SAVE_DIR}\n")
    except Exception as e:
        log_failure(e, logger)
    finally:
        if (args.sensitivity):
            run_sensitivity_analysis(pipeline=True)
        # Warn user if running out of log file space
        check_log_rotation_limits(LOG_FILEPATH)

def extract_data(force_extract):
    if (not has_totals_csv() or force_extract):
        extract_all_data(args.folder)
        return True
    else:
        logger.warning("Skipping data extraction. Using saved data instead.")
        return False


def log_failure(e, logger):
        logger.critical(f"\nExiting data pipeline due to error:")
        logger.critical(f"{e}", exc_info=True)
        logger.critical(f"\n❌ Data pipeline failed ❌")
        logger.critical(f"\nSee logs saved to {LOG_FILEPATH}\n")

def init_pipeline(args, logger, sens):
    logger.debug("Initializing pipeline")
    create_dirs(sens)
    init_extract()
    init_process()
    init_analyze()
    set_weights(args.weights)
    if sens:
        init_sensitivity(args, pipeline=True)

'''
    Run the pipeline by calling 'python3 run_pipeline.py'
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESCR)
    parser.add_argument("-f", "--folder", default=JSON_DIR, help=HELP_FOLDER)
    parser.add_argument("-s", "--show", action="store_true", help=HELP_SHOW)
    parser.add_argument("-v", "--verbose", action="store_true", help=HELP_VERBOSE)
    parser.add_argument("-d", "--debug", action="store_true", help=HELP_DEBUG)
    parser.add_argument("-q", "--quiet", action="store_true", help=HELP_QUIET)
    parser.add_argument("-n","--sensitivity", action="store_true", help=HELP_SENSITIVITY)
    parser.add_argument("-w","--weights", default=WEIGHT_SCHEME, help=HELP_WEIGHTS)
    parser.add_argument("-c", "--compact", action="store_true", help=HELP_COMPACT)
    args = parser.parse_args()

    logger = init_logging(LOG_FILEPATH, args.verbose, args.debug, args.quiet)
    logger.propagate = False
    set_logger(logger)
    logger.debug(f"\nArgs: {args}\n")

    init_pipeline(args, logger, args.sensitivity)
    run_full_pipeline(args, logger)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
