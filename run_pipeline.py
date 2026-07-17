import argparse
from datetime import datetime

import config

from src.utils import init_logging, set_logger, create_dirs, set_weights, check_log_limits, log_pipeline_failure
from src.extract_data import init_extract, extract_data
from src.process_data import init_process, process_data
from src.analyze_data import init_analyze, analyze_data
from src.sensitivity import init_sensitivity, run_sensitivity_analysis


def run_full_pipeline(args, logger):
    logger.warning("\n⏳ Begin running data pipeline")
    try:
        extracted_pkg = extract_data(args.folder)
        processed_pkg = process_data(extracted_pkg)
        analyze_data(processed_pkg, args.show, args.compact)

        logger.warning("🎉 Data pipeline completed successfully.")
        logger.warning(f"📁 See data and analysis in {config.SAVE_DIR}\n")
        if (args.sensitivity):
            sensitivity_pkg = get_sensitivity_data(extracted_pkg, processed_pkg)
            run_sensitivity_analysis(sensitivity_pkg)
    except Exception as e:
        log_pipeline_failure(e, logger)
    finally:
        # Warn user if running out of log file space
        check_log_limits(config.LOG_FILEPATH)#


def setup_args():
    parser = argparse.ArgumentParser(prog=config.PROG_NAME, description=config.PROG_DESCR)
    parser.add_argument("-f", "--folder", default=config.JSON_DIR, help=config.HELP_FOLDER)
    parser.add_argument("-s", "--show", action="store_true", help=config.HELP_SHOW)
    parser.add_argument("-v", "--verbose", action="store_true", help=config.HELP_VERBOSE)
    parser.add_argument("-d", "--debug", action="store_true", help=config.HELP_DEBUG)
    parser.add_argument("-q", "--quiet", action="store_true", help=config.HELP_QUIET)
    parser.add_argument("-n","--sensitivity", action="store_true", help=config.HELP_SENSITIVITY)
    parser.add_argument("-w","--weights", default=config.WEIGHT_SCHEME, help=config.HELP_WEIGHTS)
    parser.add_argument("-c", "--compact", action="store_true", help=config.HELP_COMPACT)
    return parser.parse_args()


def init_pipeline(args, logger):
    logger.debug("Initializing pipeline")
    logger.debug(f"\nArgs: {args}\n")
    create_dirs(sens=args.sensitivity)
    init_extract()
    init_process()
    init_analyze()
    set_weights(args.weights)
    if args.sensitivity:
        init_sensitivity(args)


def get_sensitivity_data(extracted, processed):
    return {
            "scenario_names": extracted["scenario_names"],
            "attributes_raw": extracted["attributes_data"],
            "attributes_norm": processed["attributes_norm"]
            }


def clear_loggers(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


'''
 Run the pipeline by calling 'python3 run_pipeline.py'
'''
if __name__ == "__main__":
    args = setup_args()
    logger = init_logging(config.LOG_FILEPATH, args.verbose, args.debug, args.quiet)
    set_logger(logger)
    init_pipeline(args, logger)


    run_full_pipeline(args, logger)
    clear_loggers(logger)
