import subprocess
import os
import shutil
from datetime import datetime
import requests
import json

def get_stable_version(package_name):
    """Get the latest stable version of a package using PyPI JSON API."""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        response.raise_for_status()
        data = response.json()
        return data["info"]["version"]
    except requests.RequestException as e:
        print(f"Error searching for package {package_name}: {e}")
        return None
    except KeyError:
        print(f"Could not find stable version for package {package_name}")
        return None

def update_requirements_file():
    requirements_file = 'requirements.txt'
    history_dir = 'requirements_history'
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    try:
        with open(requirements_file, 'r') as file:
            lines = file.readlines()

        old_file_path = os.path.join(history_dir, f'requirements_{timestamp}_old.txt')
        new_file_path = os.path.join(history_dir, f'requirements_{timestamp}_new.txt')

        with open(old_file_path, 'w') as file:
            file.writelines(lines)

        updated_lines = []
        for line in lines:
            if '==' in line:
                package_name = line.split('==')[0].strip()
                stable_version = get_stable_version(package_name)
                if stable_version:
                    updated_lines.append(f'{package_name}=={stable_version}\n')
                else:
                    updated_lines.append(line)
            else:
                # Handle lines with URLs or paths
                updated_lines.append(line)

        with open(new_file_path, 'w') as file:
            file.writelines(updated_lines)

        shutil.copy(new_file_path, requirements_file)
        print("requirements.txt has been updated with stable versions.")
        print(f"Old requirements saved to {old_file_path}")
        print(f"New requirements saved to {new_file_path}")
    except FileNotFoundError:
        print("requirements.txt not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_requirements_file()
