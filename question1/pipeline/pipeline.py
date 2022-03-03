
import features
import constants
import filepaths

import os
from absl import logging
import tensorflow as tf

import pandas as pd

from tfx import v1 as tfx
from tfx.v1.components import CsvExampleGen 
from tfx.v1.components import StatisticsGen 
from tfx.v1.components import SchemaGen
from tfx.v1.components import ExampleValidator
from tfx.v1.components import Transform
from tfx.v1.components import Trainer
from tfx.v1.components import Evaluator
from tfx.v1.components import Pusher
from tfx.v1.components import Tuner

from tfx.v1 import proto
from tfx.proto import trainer_pb2

import tensorflow_model_analysis as tfma

def create_pipeline(pipeline_name: str, pipeline_root: str, 
                    metadata_path: str, serving_model_dir: str) -> tfx.dsl.Pipeline:

    components = [] # List to contain all the components

    # Component to load the data
    input = proto.Input(splits=[
        proto.Input.Split(name='train', pattern='train_data.csv'),
        proto.Input.Split(name='eval', pattern='val_data.csv'),
    ])   

    example_gen = CsvExampleGen(input_base=filepaths._segregated_base, input_config=input)
    components.append(example_gen)

    # Component to generate the statistics
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # Component to generate the schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )
    components.append(schema_gen)

    # Component to validate the data
    example_validator = ExampleValidator(
                statistics=statistics_gen.outputs['statistics'],
                schema=schema_gen.outputs['schema']
            )
    components.append(example_validator)

    # Component to preprocess the data
    transform = Transform(
                    examples=example_gen.outputs['examples'],
                    schema=schema_gen.outputs['schema'],
                    module_file=filepaths._transform_module_file
                )
    components.append(transform)

    # Component to tune the model
    tuner = Tuner(
        module_file=filepaths._trainer_module_file,  
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=20),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=5))
    components.append(tuner)

    # Component to train the model
    trainer = Trainer(
                module_file=filepaths._trainer_module_file,
                examples=transform.outputs['transformed_examples'],
                transform_graph=transform.outputs['transform_graph'],
                schema=schema_gen.outputs['schema'],
                hyperparameters = tuner.outputs['best_hyperparameters'],
                train_args=tfx.proto.TrainArgs(num_steps=constants.TRAINING_STEPS),
                eval_args=tfx.proto.EvalArgs(num_steps=constants.VALIDATION_STEPS),
            )
    components.append(trainer)

    # Component to obtain the latest 'blessed' model for validation
    model_resolver = tfx.dsl.Resolver(
                        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
                        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
                        model_blessing=tfx.dsl.Channel(
                            type=tfx.types.standard_artifacts.ModelBlessing
                        )
                    ).with_id('latest_blessed_model_resolver')
    components.append(model_resolver)

    # Component to validate the trained model 
    model_specs = [tfma.ModelSpec(signature_name='serving_default', 
                                preprocessing_function_names=['tft_layer'],
                                label_key=features.transformed_name(features.LABEL_KEY))]

    slicing_specs = [tfma.SlicingSpec()]

    metrics_specs = [tfma.MetricsSpec(metrics=[
                                        tfma.MetricConfig(class_name='AUC'),
                                        tfma.MetricConfig(class_name='SparseCategoricalCrossentropy'),
                                        tfma.MetricConfig(
                                            class_name='SparseCategoricalAccuracy',
                                            threshold=tfma.MetricThreshold(
                                                value_threshold=tfma.GenericValueThreshold(lower_bound={'value': constants.LOWER_BOUND}),
                                                change_threshold=tfma.GenericChangeThreshold(
                                                    direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                                    absolute={'value': -1e-10})))
                                        ]      
                                    ),
                    ]

    eval_config = tfma.EvalConfig(
        model_specs=model_specs,
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs,
    )

    evaluator = Evaluator(
                        examples=example_gen.outputs['examples'],
                        model=trainer.outputs['model'],
                        baseline_model=model_resolver.outputs['model'],
                        eval_config=eval_config)
    components.append(evaluator)

    pusher = Pusher(
                model=trainer.outputs['model'],
                model_blessing=evaluator.outputs['blessing'],
                push_destination=tfx.proto.PushDestination(
                    filesystem=tfx.proto.PushDestination.Filesystem(
                        base_directory=filepaths._serving_model_dir
                    )
                )
            )
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
                    .sqlite_metadata_connection_config(metadata_path),
        components=components)
