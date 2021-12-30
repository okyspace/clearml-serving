import re
import subprocess
import json
from argparse import ArgumentParser
from time import time
from typing import Optional

from pathlib2 import Path

from clearml import Task, Logger
from clearml.backend_api.utils import get_http_session_with_retry
# from clearml_serving.serving_service import ServingService
from serving_service import ServingService
from EndPoint import EndPoint

import shutil
from clearml import Task, Model, InputModel


class TritonProxy(object):
    _metric_line_parsing = r"(\w+){(gpu_uuid=\"[\w\W]*\",)?model=\"(\w+)\",\s*version=\"(\d+)\"}\s*([0-9.]*)"
    _default_metrics_port = 8002
    _config_pbtxt_section = 'config.pbtxt'

    def __init__(
            self,
            args,  # Any
            task,  # type: Task
            serving_id,  # type: str
            metric_host=None,  # type: Optional[str]
            metric_port=None,  # type: int
    ):
        # type: (...) -> None
        self._http_session = get_http_session_with_retry()
        self.args = dict(**args.__dict__) if args else {}
        self.task = task
        self.remote_logger = task.get_logger() or None
        self.serving_id = serving_id
        self.metric_host = metric_host or '0.0.0.0'
        self.metric_port = metric_port or self._default_metrics_port
        self._parse_metric = re.compile(self._metric_line_parsing)
        self._timestamp = time()

    # get triton metrics, parse it, report scalar
    def report_metrics(self, remote_logger):
        # type: (Optional[Logger]) -> bool
        # iterations are seconds from start
        iteration = int(time() - self._timestamp)
        print('Report metrics, timestamp {}'.format(self._timestamp))

        report_msg = "Update from Triton Proxy ::: reporting metrics: relative time {} sec".format(iteration)
        # KY why need to report text to both task and remote; if task id same, still need?
        # log in clearml console that metrics is being reported
        # self.task.get_logger().report_text(report_msg)
        if remote_logger:
            remote_logger.report_text(report_msg)

        # noinspection PyBroadException
        # get triton metrics
        try:
            request = self._http_session.get('http://{}:{}/metrics'.format(
                self.metric_host, self.metric_port))
            if not request.ok:
                return False
            content = request.content.decode().split('\n')
        except Exception:
            return False

        # got thru lines in triton metrics
        for line in content:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # noinspection PyBroadException
            try:
                metric, gpu_uuid, variant, version, value = self._parse_metric.match(line).groups()
                value = float(value)
                # print('metric: {} gpu_uuid {}, variant {} version {} value {}'.format(metric, gpu_uuid, variant, version, value))
            except Exception:
                continue

            # on the remote logger we add our own Task ID (unique ID),
            # to support multiple servers reporting to the same service controller
            if remote_logger:
                remote_logger.report_scalar(
                    title=metric,
                    series='{}.v{}.{}'.format(variant, version, self.task.id),
                    iteration=iteration,
                    value=value
                )
        print("Report metrics end, wait for next reporting ......\n\n")

    def maintenance_daemon(
            self,
            local_model_repo='/models',  # type: str
            update_frequency_sec=1.0,  # type: float
            metric_frequency_sec=1.0  # type: float
    ):
        # type: (...) -> None
        print("run maintenance_daemon ......")
        Path(local_model_repo).mkdir(parents=True, exist_ok=True)
        self.triton_model_service_update_step(self.task, model_repository_folder=local_model_repo)  
        # todo: log triton server outputs when running locally
        base_freq = min(update_frequency_sec, metric_frequency_sec)
        metric_tic = update_tic = time()
        while True:
            # update models; e.g. new ver
            if time() - update_tic > update_frequency_sec:
                update_tic = time()
                self.triton_model_service_update_step(self.task, model_repository_folder=local_model_repo)          

            # update stats
            if time() - metric_tic > metric_frequency_sec:
                metric_tic = time()
                self.report_metrics(self.remote_logger)

    def get_endpoints(self, task):
        endpoints = self.task._get_configuration_dict(name='endpoints')   
        endpoints = json.loads(json.dumps(endpoints))

        for url, ep in endpoints.items():
            endpoints[url] = EndPoint(
                serving_url=ep.get('serving_url'),
                model_ids=list(ep.get('model_ids')) if ep.get('model_ids') else None,
                model_name=ep.get('model_name'),
                model_project=ep.get('model_project'),
                model_tags=ep.get('model_tags'),
                max_num_revisions=ep.get('max_versions') or None,
                versions={},
                model_config_blob='',
            )

        return endpoints

    def get_endpoint_version_model_id(self, serving_url):
        # type: (str) -> Dict[int, str]
        """
        Return dict with model versions and model id for the specific serving url
        If serving url is not found, return None

        :param serving_url: sering url string

        :return: dictionary keys are the versions (integers) and values are the model IDs (str)
        """

        curr_serving_endpoints = self.task._get_configuration_dict(name='serving_state')
        model_id = curr_serving_endpoints.get(serving_url) or { serving_url: None }

        return model_id

    def triton_model_service_update_step(self, task, model_repository_folder=None, verbose=True):
        print("Triton Proxy Model Update ......")
        # type: (Optional[str], bool) -> None

        # check if something changed since last time
        # if not self.update(force=self._last_update_step is None):
        #     return

        self._last_update_step = time()

        if not model_repository_folder:
            model_repository_folder = '/models/'

        if verbose:
            print('Updating local model folder: {}'.format(model_repository_folder))

        # get endpoints created at serving service
        endpoints = self.get_endpoints(self.task)
        for url, endpoint in endpoints.items():
            folder = Path(model_repository_folder) / url
            folder.mkdir(parents=True, exist_ok=True)

            # Look for the config on the Model generated Task
            # found_models = Model.query_models(project_name=model_project, model_name=model_name, tags=model_tags) or []
            endpoint.model_config_blob = self.task.get_configuration_object(
            name="model.{}".format(endpoint.serving_url))

            with open((folder / 'config.pbtxt').as_posix(), 'wt') as f:
                f.write(endpoint.model_config_blob)

            # download model versions
            for version, model_id in self.get_endpoint_version_model_id(serving_url=url).items():
                # print('ver {}, model_id {}'.format(version, model_id))
                if model_id:
                    model_folder = folder / str(version)

                    model_folder.mkdir(parents=True, exist_ok=True)
                    model = None
                    # noinspection PyBroadException
                    try:
                        # load existing model in system
                        model = InputModel(model_id)
                        # retrieve valid link to model file(s)
                        local_path = model.get_local_copy()
                    except Exception:
                        local_path = None
                    if not local_path:
                        print("Error retrieving model ID {} []".format(model_id, model.url if model else ''))
                        continue

                    local_path = Path(local_path)

                    if verbose:
                        print('Update model v{} in {}'.format(version, model_folder))

                    # if this is a folder copy every and delete the temp folder
                    if local_path.is_dir():
                        # we assume we have a `tensorflow.savedmodel` folder
                        model_folder /= 'model.savedmodel'
                        model_folder.mkdir(parents=True, exist_ok=True)
                        # rename to old
                        old_folder = None
                        if model_folder.exists():
                            old_folder = model_folder.parent / '.old.{}'.format(model_folder.name)
                            model_folder.replace(old_folder)
                        if verbose:
                            print('copy model into {}'.format(model_folder))
                        shutil.copytree(
                            local_path.as_posix(), model_folder.as_posix(), symlinks=False,
                        )
                        if old_folder:
                            shutil.rmtree(path=old_folder.as_posix())
                        # delete temp folder
                        shutil.rmtree(local_path.as_posix())
                    else:
                        # single file should be moved
                        target_path = model_folder / local_path.name
                        old_file = None
                        if target_path.exists():
                            old_file = target_path.parent / '.old.{}'.format(target_path.name)
                            target_path.replace(old_file)
                        shutil.move(local_path.as_posix(), target_path.as_posix())
                        if old_file:
                            old_file.unlink()

        print('Upate Model Repo ends ........\n\n')

def main():
    title = 'clearml-serving - Nvidia Triton Engine Proxy'
    print(title)
    parser = ArgumentParser(prog='clearml-serving', description=title)
    parser.add_argument(
        '--serving-id', default=None, type=str, required=True,
        help='Specify main serving service Task ID')
    parser.add_argument(
        '--project', default='serving', type=str,
        help='Optional specify project for the serving engine Task')
    parser.add_argument(
        '--name', default='nvidia-triton', type=str,
        help='Optional specify task name for the serving engine Task')
    parser.add_argument(
        '--update-frequency', default=5, type=float,
        help='Model update frequency in minutes')
    parser.add_argument(
        '--metric-frequency', default=2, type=float,
        help='Metric reporting update frequency in minutes')
    parser.add_argument(
        '--t-http-port', type=str, help='<integer> The port for the server to listen on for HTTP requests')
    parser.add_argument(
        '--t-http-thread-count', type=str, help='<integer> Number of threads handling HTTP requests')
    parser.add_argument(
        '--t-allow-grpc', type=str, help='<integer> Allow the server to listen for GRPC requests')
    parser.add_argument(
        '--t-grpc-port', type=str, help='<integer> The port for the server to listen on for GRPC requests')
    parser.add_argument(
        '--t-grpc-infer-allocation-pool-size', type=str,
        help='<integer> The maximum number of inference request/response objects that remain '
             'allocated for reuse. As long as the number of in-flight requests doesn\'t exceed '
             'this value there will be no allocation/deallocation of request/response objects')
    parser.add_argument(
        '--t-pinned-memory-pool-byte-size', type=str,
        help='<integer> The total byte size that can be allocated as pinned system '
             'memory. If GPU support is enabled, the server will allocate pinned '
             'system memory to accelerate data transfer between host and devices '
             'until it exceeds the specified byte size. This option will not affect '
             'the allocation conducted by the backend frameworks. Default is 256 MB')
    parser.add_argument(
        '--t-cuda-memory-pool-byte-size', type=str,
        help='<<integer>:<integer>> The total byte size that can be allocated as CUDA memory for '
             'the GPU device. If GPU support is enabled, the server will allocate '
             'CUDA memory to minimize data transfer between host and devices '
             'until it exceeds the specified byte size. This option will not affect '
             'the allocation conducted by the backend frameworks. The argument '
             'should be 2 integers separated by colons in the format <GPU device'
             'ID>:<pool byte size>. This option can be used multiple times, but only '
             'once per GPU device. Subsequent uses will overwrite previous uses for '
             'the same GPU device. Default is 64 MB')
    parser.add_argument(
        '--t-min-supported-compute-capability', type=str,
        help='<float> The minimum supported CUDA compute capability. GPUs that '
             'don\'t support this compute capability will not be used by the server')
    parser.add_argument(
        '--t-buffer-manager-thread-count', type=str,
        help='<integer> The number of threads used to accelerate copies and other'
             'operations required to manage input and output tensor contents.'
             'Default is 0')
    parser.add_argument(
        '--metrics-ip', type=str,
        help='')

    args = parser.parse_args()

    # print('serving id {}'.format(args.serving_id))
    # print('metrics ip {}'.format(args.metrics_ip))

    # get clearml task using serving id
    task = Task.get_task(task_id=args.serving_id)

    helper = TritonProxy(args, task, metric_host=args.metrics_ip, serving_id=args.serving_id)

    # this function will never end
    helper.maintenance_daemon(
        local_model_repo='/models',
        update_frequency_sec=args.update_frequency*1.0,
        metric_frequency_sec=args.metric_frequency*1.0,
    )


if __name__ == '__main__':
    print("Starting Triton Proxy ....")
    main()
