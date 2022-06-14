import os
import sys

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)

# name: The name of this test suite.
config.name = 'TFLITE'

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

#config.use_default_substitutions()
config.excludes = [
    'coco_data.py',
    'imagenet_data.py',
    'lit.cfg.py',
    'lit.site.cfg.py',
    'squad_data.py',
]

config.substitutions.extend([
    ('%PYTHON', sys.executable),
])

config.environment['PYTHONPATH'] = ":".join(sys.path)

project_root = os.path.dirname(os.path.dirname(__file__))

# Enable features based on -D FEATURES=hugetest,vulkan
# syntax.
features_param = lit_config.params.get('FEATURES')
if features_param:
    config.available_features.update(features_param.split(','))
