import argparse

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
from tfx.utils import channel

_IMAGE = "gcr.io/cloud-ml-pipelines-test/tfx-kfp-runner"

_QUERY = ("SELECT "
          "    pickup_community_area, "
          "    fare, "
          "    EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month, "
          "    EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour, "
          "    EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day, "
          "    UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp, "
          "    pickup_latitude, "
          "    pickup_longitude, "
          "    dropoff_latitude, "
          "    dropoff_longitude, "
          "    trip_miles, "
          "    pickup_census_tract, "
          "    dropoff_census_tract, "
          "    payment_type, "
          "    company, "
          "    trip_seconds, "
          "    dropoff_community_area, "
          "    tips "
          "  FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` "
          "  WHERE RAND() < 0.000001 ")


class TfxComponentWrapper(dsl.ContainerOp):

  @classmethod
  def setup_pipeline_params(cls, output_dir, project_id, gcp_region,
                            beam_runner):
    cls._output_dir = output_dir
    cls._project_id = project_id
    cls._gcp_region = gcp_region
    cls._beam_runner = beam_runner

  def __init__(self,
               component: base_component.BaseComponent,
               input_dict: Optional[Dict] = None):
    output_dict = dict(
        (k, v.get()) for k, v in component.outputs.get_all().items())

    outputs = output_dict.keys()
    file_outputs = {
        output: '/output/ml_metadata/{}'.format(output) for output in outputs
    }

    arguments = [
        '--output_dir',
        self._output_dir,
        '--project_id',
        self._project_id,
        '--gcp_region',
        self._gcp_region,
        '--beam_runner',
        self._beam_runner,
        component.component_name,
    ]

    if input_dict:
      for k, v in input_dict.items():
        if isinstance(v, float) or isinstance(v, int):
          v = str(v)
        arguments.append('--{}'.format(k))
        arguments.append(v)

    super().__init__(
        name=component.component_name,
        image=_IMAGE,
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

  def __init__(self, query: dsl.PipelineParam):
    component = big_query_example_gen_component.BigQueryExampleGen('')
    super().__init__(component, {"query": query})


class StatisticsGen(TfxComponentWrapper):

  def __init__(self, input_data: dsl.PipelineParam):
    component = statistics_gen_component.StatisticsGen(
        channel.Channel('ExamplesPath'))
    super().__init__(component, {"input_data": input_data})


class SchemaGen(TfxComponentWrapper):

  def __init__(self, stats: dsl.PipelineParam):
    component = schema_gen_component.SchemaGen(
        channel.Channel('ExampleStatisticsPath'))
    super().__init__(component, {"stats": stats})


class ExampleValidator(TfxComponentWrapper):

  def __init__(self, stats: str, schema: str):
    component = example_validator_component.ExampleValidator(
        channel.Channel('ExampleStatisticsPath'), channel.Channel('SchemaPath'))

    super().__init__(component, {"stats": stats, "schema": schema})


class Transform(TfxComponentWrapper):

  def __init__(self, input_data: dsl.PipelineParam, schema: dsl.PipelineParam,
               module_file: dsl.PipelineParam):
    component = transform_component.Transform(
        input_data=channel.Channel('ExamplesPath'),
        schema=channel.Channel('SchemaPath'),
        module_file='')

    super().__init__(component, {
        "input_data": input_data,
        "schema": schema,
        "module_file": module_file,
    })


@dsl.pipeline(
    name="Chicago Taxi Cab Tip Prediction Pipeline", description="TODO")
def pipeline(
    project_id=dsl.PipelineParam(name="GCP Project ID", value='my-gcp-project'),
    output_dir=dsl.PipelineParam(
        name='Base output directory', value='gs://my-bucket'),
    gcp_region=dsl.PipelineParam(
        name='GCP Region for Dataflow', value='us-central1'),
    beam_runner=dsl.PipelineParam(
        name='Beam Runner to use', value='DataflowRunner'),
    query=dsl.PipelineParam(name='BigQuery query', value=_QUERY),
    module_file=dsl.PipelineParam(
        name='Module File', value='gs://my-module-file'),
):
  TfxComponentWrapper.setup_pipeline_params(
      output_dir=output_dir,
      project_id=project_id,
      gcp_region=gcp_region,
      beam_runner=beam_runner)

  example_gen = BigQueryExampleGen(query=query)
  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

  infer_schema = SchemaGen(stats=statistics_gen.outputs['output'])

  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs['output'],
      schema=infer_schema.outputs['output'])

  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=infer_schema.outputs['output'],
      module_file=module_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Chicago Taxi Cab Pipeline")
  parser.add_argument("--filename", type=str)
  args = parser.parse_args()

  fname = args.filename if args.filename else __file__

  compiler.Compiler().compile(pipeline, fname + '.tar.gz')
