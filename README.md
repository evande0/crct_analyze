# CRCT Analyze
Analyzes performance data of different Climate Resilient Cities Tool projects using their JSON files. 

# Quick Run
Call `python run_pipeline.py` from the project root to run the pipeline on the JSON files under `data/project_JSON`

# Using the Pipeline
1) Clone the repository: `git clone https://github.com/evande0/crct_analyze.git`
2) Install dependencies:  `pip install -r requirements.txt`
3) Save CRCT projects as JSON files to `data/project_json` or use the included samples
4) [Optional] Open `config.yaml` to customize constants related to directories, weighting schemes, sensitivity analysis, and logging. Warning: Changing constants in `config.py` could break the pipeline.
5) [Optional] View available flags by calling `run_pipeline.py --help`
6) Run the pipeline: `python run_pipeline.py`
7) Review outputs in the `results/[timestamp]` folder

# Pipeline Outputs
- Logs are appended to `logs/pipeline.log`. Logfiles are rotated upon reaching max size (see `config.yaml`)
- Each run generates outputs in a timestamped folder under `results/`
- Extracted scenario data is saved in CSV format under `results/[timestamp]/raw`
- Normalized and weighted attribute data is saved in CSV format under `results/[timestamp]/processed`
- Visualizations are saved as PNGs under `results/[timestamp]/viz`
- Sensitivity analysis data is saved as a CSV and sensitivity curves are saved as PNGs under `results/[timestamp]/sensitivity 