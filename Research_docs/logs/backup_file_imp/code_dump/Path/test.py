import ruamel.yaml
from path_manager_with_root import PathManager  # Assuming PathManager class is in a file named PathManager.py

def test_path_conversion():
    # Create a PathManager instance
    path_manager = PathManager(r"C:\Users\siu856522160\major\Research_docs\Path\path.yaml")

    # Test case for Linux to Windows path conversion
    linux_paths = {
        'root_path': '/path/to/project',
        'config_files': [
            {'config_dir': 'config'},
            {'settings_file': 'config/settings.yaml'}
        ],
        'data_files': [
            {'data_dir': 'data'},
            {'models_dir': 'models'}
        ],
        'logs': [
            {'logs_dir': 'logs'},
            {'error_log_file': 'logs/error.log'}
        ],
        'new_dir': 'NewDirectory',
        'source_code': [
            {'src_dir': 'src'},
            {'tests_dir': 'tests'}
        ]
    }

    path_manager.paths = linux_paths
    path_manager.convert_and_save_paths(target_os='nt')
    print(path_manager)
    print(path_manager.paths['root_path'])
    # Verify Linux paths are converted to Windows paths
    assert path_manager.paths['root_path'] == 'C:\\path\\to\\project'
    assert path_manager.paths['config_files'][0]['config_dir'] == 'config'
    assert path_manager.paths['config_files'][1]['settings_file'] == 'config\\settings.yaml'
    assert path_manager.paths['data_files'][0]['data_dir'] == 'data'
    assert path_manager.paths['data_files'][1]['models_dir'] == 'models'
    assert path_manager.paths['logs'][0]['logs_dir'] == 'logs'
    assert path_manager.paths['logs'][1]['error_log_file'] == 'logs\\error.log'
    assert path_manager.paths['new_dir'] == 'NewDirectory'
    assert path_manager.paths['source_code'][0]['src_dir'] == 'src'
    assert path_manager.paths['source_code'][1]['tests_dir'] == 'tests'

    # Test case for Windows to Linux path conversion
    windows_paths = {
        'root_path': 'C:\\path\\to\\project',
        'config_files': [
            {'config_dir': 'config'},
            {'settings_file': 'config\\settings.yaml'}
        ],
        'data_files': [
            {'data_dir': 'data'},
            {'models_dir': 'models'}
        ],
        'logs': [
            {'logs_dir': 'logs'},
            {'error_log_file': 'logs\\error.log'}
        ],
        'new_dir': 'NewDirectory',
        'source_code': [
            {'src_dir': 'src'},
            {'tests_dir': 'tests'}
        ]
    }

    path_manager.paths = windows_paths
    path_manager.convert_and_save_paths(target_os='posix')

    # Verify Windows paths are converted to Linux paths
    assert path_manager.paths['root_path'] == '/path/to/project'
    assert path_manager.paths['config_files'][0]['config_dir'] == 'config'
    assert path_manager.paths['config_files'][1]['settings_file'] == 'config/settings.yaml'
    assert path_manager.paths['data_files'][0]['data_dir'] == 'data'
    assert path_manager.paths['data_files'][1]['models_dir'] == 'models'
    assert path_manager.paths['logs'][0]['logs_dir'] == 'logs'
    assert path_manager.paths['logs'][1]['error_log_file'] == 'logs/error.log'
    assert path_manager.paths['new_dir'] == 'NewDirectory'
    assert path_manager.paths['source_code'][0]['src_dir'] == 'src'
    assert path_manager.paths['source_code'][1]['tests_dir'] == 'tests'

    print("All path conversion tests passed!")

if __name__ == "__main__":
    test_path_conversion()
