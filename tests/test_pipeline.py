from nexus.generator import generate_data
from nexus.pipeline import run_pipeline

def test_pipeline_creates_outputs(tmp_path):
    generate_data(1500,1,str(tmp_path/"data/bronze"))
    import os
    old=os.getcwd()
    try:
        os.chdir(tmp_path)
        result=run_pipeline()
        assert len(result["transactions"])==1500
        assert (tmp_path/"data/gold/daily_metrics.csv").exists()
        assert result["quality"]["quality_score"]==100
    finally:
        os.chdir(old)
