

# PathManager Documentation

## Overview

The `PathManager` class is designed to manage and convert file paths between different operating systems (Windows and Linux). It loads paths from a YAML configuration file, converts them as needed, and saves the updated paths back to the file.

## Features

- Load paths from a YAML configuration file.
- Convert paths between Windows and Linux formats.
- Save the updated paths back to the configuration file.
- Retrieve specific paths based on keys.
- Set new paths and update the configuration file.

## YAML Configuration Format

The YAML configuration file should follow this format:

```yaml
root_path: /path/to/project

source_code:
    - src_dir: src
    - tests_dir: tests

data_files:
    - data_dir: data
    - models_dir: models

config_files:
    - config_dir: config
    - settings_file: config/settings.yaml

logs:
    - logs_dir: logs
    - error_log_file: logs/error.log
new_dir: NewDirectory
```

# Reference:
1. https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python

2. https://stackoverflow.com/questions/25108581/python-yaml-dump-bad-indentation