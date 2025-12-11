# Twitter-Stock-Forecaster

To run the system, first do "python src/ingest_history.py" to to pre-process the raw dataset and initialize the mysql lite db.
Afterwards, run "python -m src.main_orchestrator" to run the pipeline in root directory.

The system flows from main_orchestrator.py to run the schedule scraper.py, then runs inference.
