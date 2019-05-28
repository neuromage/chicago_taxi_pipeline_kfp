# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import subprocess

from typing import Optional, Dict, List

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

_COMMAND = [
    'python',
    '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py',
]


class TfxComponentRunner(object):

  def __init__(self, args, component, input_dict=None):
    executor_class_path = '.'.join(
        [component.executor.__module__, component.executor.__name__])

    output_dict = dict(
        (k, v.get()) for k, v in component.outputs.get_all().items())

    beam_pipeline_args = [
        '--experiments=shuffle_mode=auto',
        '--runner={}'.format(args.beam_runner),
        '--project={}'.format(args.project_id),
        '--temp_location={}'.format(os.path.join(args.output_dir, 'tmp')),
        '--region={}'.format(args.gcp_region),
    ]

    exec_properties = {
        "beam_pipeline_args": beam_pipeline_args,
        'output_dir': args.output_dir,
    }

    exec_properties.update(component.exec_properties)

    self._command = [
        'python',
        '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py',
        '--exec_properties',
        json.dumps(exec_properties),
        '--outputs',
        types.jsonify_tfx_type_dict(output_dict),
        '--executor_class_path',
        executor_class_path,
        component.component_name,
    ]

    if input_dict:
      for k, v in input_dict.items():
        if isinstance(v, float) or isinstance(v, int):
          v = str(v)
        self._command.append('--{}'.format(k))
        self._command.append(v)

  def run(self):
    print('executing: {}'.format(self._command))
    subprocess.check_call(self._command)


class BigQueryExampleGenRunner(TfxComponentRunner):

  def __init__(self, args):
    component = big_query_example_gen_component.BigQueryExampleGen(args.query)
    super(BigQueryExampleGenRunner, self).__init__(args, component)


class StatisticsGenRunner(TfxComponentRunner):

  def __init__(self, args):
    component = statistics_gen_component.StatisticsGen(
        channel.Channel('ExamplesPath'))
    super(StatisticsGenRunner, self).__init__(args, component,
                                              {"input_data": args.input_data})


class SchemaGenRunner(TfxComponentRunner):

  def __init__(self, args):
    stats = tfx.Types()
    stats.uri = args.uri
    component = schema_gen_component.SchemaGen(
        channel.Channel('ExampleStatisticsPath', static_artifact_collection=[types.TfxType('ExamplesStatisticsPath')]))
    super(SchemaGenRunner, self).__init__(component, {"stats": types.jsonify( ))']'


class ExampleValidatorRunner(TfxComponentRunner):

  def __init__(self, args):
    component = example_validator_component.ExampleValidator(
        channel.Channel('ExampleStatisticsPath'), channel.Channel('SchemaPath'))

    super().__init__(component, {"stats": args.stats, "schema": args.schema})


class TransformRunner(TfxComponentRunner):

  def __init__(self, args):
    component = transform_component.Transform(
        input_data=channel.Channel('ExamplesPath'),
        schema=channel.Channel('SchemaPath'),
        module_file=args.module_file)

    super().__init__(component, {
        "input_data": args.input_data,
        "schema": args.schema,
    })


def main():
  print('hello')
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str, required=True)
  parser.add_argument('--log_root', type=str, default='/var/tmp/log')
  parser.add_argument('--project_id', type=str, default='')
  parser.add_argument('--gcp_region', type=str, default='')
  parser.add_argument('--beam_runner', type=str, default='DirectRunner')

  subparsers = parser.add_subparsers(dest='component')

  subparser = subparsers.add_parser('BigQueryExampleGen')
  subparser.set_defaults(component=BigQueryExampleGenRunner)
  subparser.add_argument('--query', type=str, required=True)

  subparser = subparsers.add_parser('StatisticsGen')
  subparser.set_defaults(component=StatisticsGenRunner)
  subparser.add_argument('--input_data', type=str, required=True)

  subparser = subparsers.add_parser('SchemaGen')
  subparser.set_defaults(component=SchemaGenRunner)
  subparser.add_argument('--stats', type=str, required=True)

  subparser = subparsers.add_parser('ExampleValidator')
  subparser.set_defaults(component=ExampleValidatorRunner)
  subparser.add_argument('--stats', type=str, required=True)
  subparser.add_argument('--scheme', type=str, required=True)

  subparser = subparsers.add_parser('Transform')
  subparser.set_defaults(component=TransformRunner)
  subparser.add_argument('--input_data', type=str, required=True)
  subparser.add_argument('--scheme', type=str, required=True)
  subparser.add_argument('--module_file', type=str, required=True)

  args = parser.parse_args()
  component = args.component(args)
  component.run()


if __name__ == "__main__":
  main()
