import os
import ruamel.yaml

class PathManager:
    _instance = None
    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=4, offset=4)
    
    def __new__(cls, yaml_file='paths.yaml'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.paths_data = cls._instance.load_paths_from_yaml(yaml_file)
        return cls._instance
    
    def __init__(self, config_file=None):
        if config_file is None:
            # Use paths.yaml in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, 'path.yaml')
        self.config_file = config_file
        self.paths = self.load_paths()
        self.project_root = self.paths.get('root_path', '.')
        self.convert_and_save_paths(target_os=os.name)


    def __str__(self):
        paths_info = "\n".join([f"{key}: {value}" for key, value in self.paths.items()])
        return f"PathManager Configuration:\n{paths_info}"   
    
    def load_paths_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as file:
            paths_data = self.yaml.load(file)
        return paths_data
    
    def load_paths(self):
        with open(self.config_file, 'r') as file:
            yaml = ruamel.yaml.YAML()
            yaml.indent(sequence=4, offset=4)
            return yaml.load(file)

    def save_paths(self):
        with open(self.config_file, 'w') as file:
            yaml = ruamel.yaml.YAML()
            yaml.indent(sequence=4, offset=4)
            yaml.dump(self.paths, file)

    def get_path(self, key):
        path = self.paths.get(key)
        return os.path.join(self.project_root, path) if path else None

    def set_path(self, key, value):
        self.paths[key] = value
        self.save_paths()

    def convert_path(self, path, target_os):
        if target_os == 'nt':
            return path.replace('/c', 'C:').replace('/', '\\')
        else:
            return path.replace('C:', '/c').replace('\\', '/')
        
    def convert_and_save_paths(self, target_os):
        paths = self.load_paths()
        
        for key, value in paths.items():
            if isinstance(value, list):
                converted_list = []
                for item in value:
                    if isinstance(item, dict):
                        converted_item = {k: self.convert_path(v, target_os) for k, v in item.items()}
                        converted_list.append(converted_item)
                    else:
                        converted_list.append(self.convert_path(item, target_os))
                paths[key] = converted_list
            else:
                paths[key] = self.convert_path(value, target_os)
        
        self.paths = paths  # Update the paths attribute with the converted paths
        self.save_paths()   # Save the updated paths to the YAML file





    def get_project_root(self):
        return self.paths.get('root_path')

    def get_key_for_path(self, path):
        for key, value in self.paths.items():
            if value == path:
                return key
        return None
    
    def get_full_path(self, key):
        value = self.paths.get(key)
        if value:
            return os.path.join(self.project_root, value)
        else:
            return None
