import argparse
from datetime import datetime

from config import *
from pipeline_logger import *
from utils import *
from extract_data import *
from process_data import *
from analyze_data import *

'''
TODO
- Rotating console log
-
'''


'''
    Run the pipeline by calling 'python3 run_pipeline.py'
'''
def run_full_pipeline(args, logger):
    logger.warning("\n⏳ Begin running data pipeline\n")
    try:
        extract_all_data(args.folder)
        logger.warning("✅ Data extraction complete.")

        process_data()
        logger.warning("✅ Processing complete.")

        analyze_data(args.showradar)
        logger.warning("✅ Analysis complete.")

        logger.warning("\n🎉 Data pipeline completed successfully.")
        logger.warning(f"\nSee pipeline artifacts in {SAVE_DIR}\n")
    except Exception as e:
        log_failure(e, logger)


def log_failure(e, logger):
        logger.critical(f"\nExiting data pipeline due to error: {e}")
        logger.info(f"{e}", exc_info=True)
        logger.critical(f"\n❌ Data pipeline failed ❌")
        logger.critical(f"\nSee logs saved to {LOG_FILE}\n")

def init_pipeline(logger):
    logger.debug("Initializing pipeline")
    create_dirs()
    setup_totals_file()
    init_extract()
    init_process()
    init_analyze()

'''
    Run the pipeline by calling 'python3 run_pipeline.py'
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=PROG_DESCR)
    parser.add_argument("-f", "--folder", default=PROJ_DIR, help=HELP_FOLDER)
    parser.add_argument("-sr", "--showradar", action="store_true", help=HELP_SHOWRADAR)
    parser.add_argument("-v", "--verbose", action="store_true", help=HELP_VERBOSE)
    parser.add_argument("-d", "--debug", action="store_true", help=HELP_DEBUG)
    parser.add_argument("-q", "--quiet", action="store_true", help=HELP_QUIET)
    args = parser.parse_args()

    consolidate_logs(args.verbose)
    logger = init_logging(LOG_FILE, args.verbose, args.debug, args.quiet)
    logger.propagate = False
    set_logger(logger)
    if get_logger() is None:
        print("config.LOGGER is still None??")
    else:
        print("Logger set successfully")
    logger.debug(f"\n{args}")

    init_pipeline(logger)
    run_full_pipeline(args, logger)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
