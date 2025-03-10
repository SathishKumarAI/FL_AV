
import os
import yaml

class PathManager:
    def __init__(self, config_file='paths.yaml'):
        self.config_file = config_file
        self.paths = self.load_paths()
        self.project_root = self.paths.get('project_root', '.')
        # self.convert_and_save_paths()

    def load_paths(self):
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def save_paths(self):
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(self.paths, file)

    def get_path(self, key):
        path = self.paths.get(key)
        return os.path.join(self.project_root, path) if path else None

    def set_path(self, key, value):
        self.paths[key] = value
        self.save_paths()

    def convert_path(self, path, target_os):
        if target_os == 'nt':
            return path.replace('/mnt/c', 'C:').replace('/', '\\')
        else:
            return path.replace('C:', '/mnt/c').replace('\\', '/')
    def __str__(self):
        paths_info = "\n".join([f"{key}: {value}" for key, value in self.paths.items()])
        return f"PathManager Configuration:\n{paths_info}"
    def convert_and_save_paths(self,target_os="nt"):
        target_os = 'nt' if os.name == 'nt' else 'posix'
        self.paths['project_root'] = self.convert_path(self.project_root, target_os)
        self.paths = {key: self.convert_path(value, target_os) for key, value in self.paths.items()}
        self.save_paths()

    def get_project_root(self):
        return self.paths.get('project_root')

    def get_data_dir(self):
        return self.get_path('data_dir')

    def get_logs_dir(self):
        return self.get_path('logs_dir')

def test_path_manager():
    test_paths = {
        'project_root': '/mnt/c/Projects/TestProject',
        'data_dir': 'Data',
        'logs_dir': 'Logs'
    }
    config_file_path = r'C:\Users\siu856522160\major\Research_docs\Path\test_paths.yaml'
    with open(config_file_path, 'w') as file:
        yaml.safe_dump(test_paths, file)
    
    path_manager = PathManager(config_file=config_file_path)

    assert path_manager.get_project_root() == test_paths['project_root']
    
    assert path_manager.get_data_dir() == os.path.join(test_paths['project_root'], test_paths['data_dir'])
    assert path_manager.get_logs_dir() == os.path.join(test_paths['project_root'], test_paths['logs_dir'])

    print(path_manager)

    original_os_name = os.name
    os.name = 'nt'  # Simulate Windows
    path_manager.convert_and_save_paths(target_os=os.name)

    assert path_manager.get_project_root() == 'C:\\Projects\\TestProject'
    # print(path_manager)
    # print(path_manager.get_data_dir())
    assert path_manager.get_data_dir() == os.path.join(test_paths['project_root'], test_paths['data_dir'])
    assert path_manager.get_logs_dir() == os.path.join(test_paths['project_root'], test_paths['logs_dir'])
    print(path_manager)
    os.name = 'posix'  # Simulate Linux
    path_manager.convert_and_save_paths(target_os=os.name)

    assert path_manager.get_project_root() == '/mnt/c/Projects/TestProject'
    assert path_manager.get_data_dir() == os.path.join(test_paths['project_root'], test_paths['data_dir'])
    assert path_manager.get_logs_dir() == os.path.join(test_paths['project_root'], test_paths['logs_dir'])
    print(path_manager)
    os.name = original_os_name

    print("All tests passed!")

if __name__ == "__main__":
    test_path_manager()
