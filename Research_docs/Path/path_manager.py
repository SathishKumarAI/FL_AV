import os
import yaml


class PathManager:
    def __init__(self, config_file='paths.yaml'):
        self.config_file = config_file
        self.paths = self.load_paths()
        self.convert_and_save_paths()

    def load_paths(self):
        try:
            with open(self.config_file, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"Config file '{self.config_file}' not found. Creating new.")
            return {}

    def save_paths(self):
        try:
            with open(self.config_file, 'w') as file:
                yaml.safe_dump(self.paths, file)
        except Exception as e:
            print(f"Error saving paths to '{self.config_file}': {e}")

    def get_path(self, key):
        return self.paths.get(key)

    def set_path(self, key, value):
        self.paths[key] = value
        self.save_paths()

    def get_path_key(self, path):
        for key, value in self.paths.items():
            if value == path:
                return key
        return None

    def convert_path(self, path, target_os):
        if target_os == 'nt':
            return path.replace('/mnt/c', 'C:').replace('/', '\\')
        elif target_os == 'posix':
            return path.replace('C:', '/mnt/c').replace('\\', '/')

    def convert_and_save_paths(self):
        target_os = 'nt' if os.name == 'nt' else 'posix'
        self.paths = {key: self.convert_path(value, target_os) for key, value in self.paths.items()}
        self.save_paths()

    def get_project_root(self):
        return self.get_path('project_root')

    def get_data_dir(self):
        return self.get_path('data_dir')

    def get_logs_dir(self):
        return self.get_path('logs_dir')


def test_path_manager():
    """
    This function tests various functionalities of the PathManager class.
    """

    # Initialize with test paths
    test_paths = {
        'project_root': '/mnt/c/Projects/TestProject',
        'data_dir': '/mnt/c/Data',
        'logs_dir': '/mnt/c/Logs'
    }

    # Create a temporary test configuration file with sample paths
    config_file_path = r'C:\Users\siu856522160\major\Research_docs\Path\test_paths.yaml'
    with open(config_file_path, 'w') as file:
        yaml.safe_dump(test_paths, file)

    path_manager = PathManager(config_file=config_file_path)

    # Test 1: Verify paths are loaded correctly from the configuration file
    assert path_manager.get_project_root() == test_paths['project_root'], "Project root path not loaded correctly"
    assert path_manager.get_data_dir() == test_paths['data_dir'], "Data directory path not loaded correctly"
    assert path_manager.get_logs_dir() == test_paths['logs_dir'], "Logs directory path not loaded correctly"

    # Test 2: Test setting a new path and retrieving it
    path_manager.set_path('new_dir', '/mnt/c/NewDirectory')
    assert path_manager.get_path('new_dir') == '/mnt/c/NewDirectory', "Setting and retrieving a new path failed"

    # Test 3: Test getting a key by path
    assert path_manager.get_path_key('/mnt/c/NewDirectory') == 'new_dir', "Getting a key by path failed"

    # Test 4: Test conversion to Windows paths
    original_os_name = os.name
    os.name = 'nt'  # Simulate Windows
    path_manager.convert_and_save_paths()

    windows_converted_paths = {
        'project_root': 'C:\\Projects\\TestProject',
        'data_dir': 'C:\\Data',
        'logs_dir': 'C:\\Logs'
    }

    assert path_manager.get_project_root() == windows_converted_paths['project_root']
    assert path_manager.get_data_dir() == windows_converted_paths['data_dir']
    assert path_manager.get_logs_dir() == windows_converted_paths['logs_dir']

    # Test 5: Test conversion back to Linux paths
    os.name = 'posix'  # Simulate Linux
    path_manager.convert_and_save_paths()

    assert path_manager.get_project_root() == test_paths['project_root']
    assert path_manager.get_data_dir() == test_paths['data_dir']
    assert path_manager.get_logs_dir() == test_paths['logs_dir']

    # Reset the os.name back to original for further tests or operations
    os.name = original_os_name

    print("All tests passed!")


if __name__ == "__main__":
    test_path_manager()
