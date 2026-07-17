# CRCT Analyze
Analyzes performance data of different Climate Resilient Cities Tool projects using their JSON files. 

# Quick Start
Call `python run_pipeline.py` from the project root. This will analyze the JSON files included under `data/project_JSON`

# Using the Pipeline
1) Clone the repository: `git clone https://github.com/evande0/crct_analyze.git`
2) Install dependencies:  `pip install -r requirements.txt`
3) Save CRCT projects as JSON files to `data/project_json` or use the included samples.
    - **Tip**: Name JSON files as `"Scenario #: Brief Description"` or similar. The file names are used as scenario names in the visualizations. 
4) [Optional] Open `config.yaml` to customize constants related to directories, weighting schemes, sensitivity analysis, and logging.
    - **Tip**: Update the `DEFAULT` or custom weight schemes in `config.yaml`
    - **Warning**: Changing constants in config.**py** could break the pipeline.
5)  Run the pipeline: `python run_pipeline.py`
    - **Tips:**
      - Use `-w CUSTOM1` to use the CUSTOM1 weight scheme
      - Use `-n` to run the sensitivity analysis
      - Call `python run_pipeline.py —help` for a full list of available flags
6) Review outputs in the `results/[timestamp]` folder

# Pipeline Outputs
- Logs are appended to `logs/pipeline.log`. Logfiles are rotated upon reaching max size (see `config.yaml`)
- Each run saves outputs in a timestamped folder under `results/`
- Extracted scenario data is saved in CSV format under `results/[timestamp]/raw`
- Normalized and weighted attribute data is saved in CSV format under `results/[timestamp]/processed`
- Visualizations are saved as PNGs under `results/[timestamp]/viz`
- Sensitivity analysis data is saved as a CSV and sensitivity curves are saved as PNGs under `results/[timestamp]/sensitivity`