import argparse
import json

from typing import Optional, Dict, List

from kfp import dsl
from kfp import gcp
from kfp.compiler import compiler
from kubernetes import client as k8s_client
from tfx.components.example_gen.big_query_example_gen import component as big_query_example_gen_component
from tfx.components.statistics_gen import component as statistics_gen_component
from tfx.components.schema_gen import component as schema_gen_component
from tfx.components.example_validator import component as example_validator_component
from tfx.components.transform import component as transform_component
from tfx.components.trainer import component as trainer_component
from tfx.components.evaluator import component as evaluator_component
from tfx.components.model_validator import component as model_validator_component
from tfx.components.pusher import component as pusher_component
from tfx.components.base import base_component
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils import types
from tfx.utils import channel

_IMAGE = 'tensorflow/tfx:0.13rc1'
_COMMAND = [
    'python',
    '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py',
]

_BEAM_PIPELINE_ARGS = {}


class TfxComponentWrapper(dsl.ContainerOp):

  def __init__(self,
               component: base_component.BaseComponent,
               input_dict: Optional[Dict] = None):
    self.component = component

    executor_class_path = '.'.join(
        [component.executor.__module__, component.executor.__name__])

    output_dict = dict(
        (k, v.get()) for k, v in component.outputs.get_all().items())

    outputs = output_dict.keys()
    file_outputs = {
        output: '/output/ml_metadata/{}'.format(output) for output in outputs
    }

    arguments = [
        '--exec_properties',
        json.dumps(component.exec_properties),
        '--outputs',
        types.jsonify_tfx_type_dict(output_dict),
        '--executor_class_path',
        executor_class_path,
        component.component_name,
    ]

    if input_dict:
      for k, v in input_dict.items():
        # if isinstance(v, float) or isinstance(v, int):
        #   v = str(v)
        arguments.append('--{}'.format(k))
        arguments.append(v)

    super().__init__(
        name=component.component_name,
        image=_IMAGE,
        command=_COMMAND,
        arguments=arguments,
        file_outputs=file_outputs,
    )
    self.apply(gcp.use_gcp_secret('user-gcp-sa'))

    field_path = "metadata.labels['workflows.argoproj.io/workflow']"
    self.add_env_variable(
        k8s_client.V1EnvVar(
            name='WORKFLOW_ID',
            value_from=k8s_client.V1EnvVarSource(
                field_ref=k8s_client.V1ObjectFieldSelector(
                    field_path=field_path))))


class BigQueryExampleGen(TfxComponentWrapper):

  def __init__(self, query: str):
    component = big_query_example_gen_component.BigQueryExampleGen(query)
    super().__init__(component)


class StatisticsGen(TfxComponentWrapper):

  def __init__(self, input_data: str):
    component = statistics_gen_component.StatisticsGen(
        channel.Channel('ExamplesPath'))
    super().__init__(component, {"input_data": input_data})


class SchemaGen(TfxComponentWrapper):

  def __init__(self, stats: str):
    component = schema_gen_component.SchemaGen(
        channel.Channel('ExampleStatisticsPath'))
    super().__init__(component, {"stats": stats})


class ExampleValidator(TfxComponentWrapper):

  def __init__(self, stats: str, schema: str):
    component = example_validator_component.ExampleValidator(
        channel.Channel('ExampleStatisticsPath'), channel.Channel('SchemaPath'))

    super().__init__(component, {"stats": stats, "schema": schema})


class Transform(TfxComponentWrapper):

  def __init__(self, input_data: str, schema: str, module_file: str):
    component = transform_component.Transform(
        input_data=channel.Channel('ExamplesPath'),
        schema=channel.Channel('SchemaPath'),
        module_file=module_file)

    super().__init__(component, {"input_data": input_data, "schema": schema})


class Trainer(TfxComponentWrapper):

  def __init__(self, module_file: str, transformed_examples: str, schema: str,
               transform_output: str, training_steps: int,
               eval_training_steps: int):
    component = trainer_component.Trainer(
        module_file=module_file,
        transformed_examples=channel.Channel('ExamplesPath'),
        schema=channel.Channel('SchemaPath'),
        transform_output=channel.Channel('TransformPath'),
        train_args=trainer_pb2.TrainArgs(num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_training_steps))

    super().__init__(
        component, {
            "transformed_examples": transformed_examples,
            "schema": schema,
            "transform_output": transform_output
        })


class Evaluator(TfxComponentWrapper):

  def __init__(self, examples: str, model_exports: str,
               feature_slicing_spec: List[List[str]]):
    slicing_spec = evaluator_pb2.FeatureSlicingSpec()
    for slice_spec in feature_slicing_spec:
      spec = slicing_spec.specs.add()
      for column in slice_spec:
        spec.column_for_slicing.append(column)

    component = evaluator_component.Evaluator(
        channel.Channel('ExamplesPath'),
        channel.Channel('ModelExportPath'),
        feature_slicing_spec=slicing_spec)

    super().__init__(component, {
        "examples": examples,
        "model_exports": model_exports,
    })


class ModelValidator(TfxComponentWrapper):

  def __init__(self, examples: str, model: str):
    component = model_validator_component.ModelValidator(
        channel.Channel('ExamplesPath'), channel.Channel('ModelExportPath'))

    super().__init__(component, {
        "examples": examples,
        "model": model,
    })


class Pusher(TfxComponentWrapper):

  def __init__(self, model_export: str, model_blessing: str,
               serving_directory: str):
    push_destination = pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_directory))

    component = pusher_component.Pusher(
        model_export=channel.Channel('ModelExportPath'),
        model_blessing=channel.Channel('ModelBlessingPath'),
        push_destination=push_destination)

    super().__init__(component, {
        "model_export": model_export,
        "model_blessing": model_blessing,
    })


_taxi_utils = ""


@dsl.pipeline(
    name="Chicago Taxi Cab Tip Prediction Pipeline", description="TODO")
def pipeline():
  example_gen = BigQueryExampleGen(query="")

  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

  infer_schema = SchemaGen(stats=statistics_gen.outputs['output'])

  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs['output'],
      schema=infer_schema.outputs['output'])

  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=infer_schema.outputs['output'],
      module_file=_taxi_utils)

  # Train using a deprecated flag.
  trainer = Trainer(
      module_file=_taxi_utils,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['output'],
      transform_output=transform.outputs['transform_output'],
      training_steps=10000,
      eval_training_steps=5000)

  model_analyzer = Evaluator(
      examples=example_gen.outputs['examples'],
      model_exports=trainer.outputs['output'],
      feature_slicing_spec=[['trip_start_hour']])

  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['output'])

  pusher = Pusher(
      model_export=trainer.outputs['output'],
      model_blessing=model_validator.outputs['blessing'],
      serving_directory="")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Chicago Taxi Cab Pipeline")
  parser.add_argument("--filename", type=str)
  args = parser.parse_args()

  fname = args.filename if args.filename else __file__

  compiler.Compiler().compile(pipeline, fname + '.tar.gz')
