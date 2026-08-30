# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os

from packaging.version import parse as parse_version

# ``pkg_resources`` was deprecated by setuptools and is no longer installed in
# a number of otherwise valid runtime environments.  VERL only needs the
# distribution lookup here, so keep a small compatibility fallback instead of
# making every tool/adapter import depend on the deprecated package.
try:
    import pkg_resources  # type: ignore
    from pkg_resources import DistributionNotFound  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    from importlib import metadata as _importlib_metadata

    class _PkgResourcesCompat:
        @staticmethod
        def get_distribution(name):
            return _importlib_metadata.distribution(name)

    pkg_resources = _PkgResourcesCompat()
    DistributionNotFound = _importlib_metadata.PackageNotFoundError

try:
    from .utils.device import is_npu_available
except ModuleNotFoundError:  # lightweight data/adapter environments
    is_npu_available = False

try:
    from .utils.logging_utils import set_basic_config
except ModuleNotFoundError:  # logging_utils imports torch in full VERL installs
    def set_basic_config(level):
        logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]


def __getattr__(name):
    """Lazily import heavyweight VERL protocol dependencies.

    Lightweight utilities (for example the TravelGym tool adapter) should be
    importable in data-preparation environments that do not install the full
    Ray/tensordict/pandas training stack.  ``from verl import DataProto`` keeps
    its normal behaviour because Python resolves the attribute through this
    hook when it is actually requested.
    """
    if name == "DataProto":
        from .protocol import DataProto

        globals()[name] = DataProto
        return DataProto
    raise AttributeError(name)

if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    import importlib

    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()

if is_npu_available:
    package_name = 'transformers'
    required_version_spec = '4.51.0'
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        installed = parse_version(installed_version)
        required = parse_version(required_version_spec)

        if not installed >= required:
            raise ValueError(f"{package_name} version >= {required_version_spec} is required on ASCEND NPU, current version is {installed}.")
    except DistributionNotFound:
        raise ImportError(
            f"package {package_name} is not installed, please run pip install {package_name}=={required_version_spec}")
