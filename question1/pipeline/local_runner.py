
import pipeline
from tfx import v1 as tfx
from absl import logging
import filepaths

import os

def start_pipelines():

    # Load the yaml config and obtain the run parameters
    PIPELINE_NAME = filepaths._pipeline_name
    PIPELINE_ROOT = filepaths._pipeline_root
    METADATA_PATH = filepaths._metadata_path
    SERVING_MODEL_DIR = filepaths._serving_model_dir

    logging.info('Starting Pipeline')
    tfx.orchestration.LocalDagRunner().run(
        pipeline.create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            metadata_path=METADATA_PATH,
            serving_model_dir=SERVING_MODEL_DIR,
        )
    )

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    start_pipelines()
