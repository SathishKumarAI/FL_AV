Sure, you can modify the script to save the old and new versions of the `requirements.txt` file into a folder (e.g., `requirements_history`) and update the main `requirements.txt` file. Here's how you can achieve this:

### Updated Python Script

1. **Create the Folder Structure:**

   Ensure there is a folder named `requirements_history` in your project directory where the script will store the old and new versions of `requirements.txt`.

2. **Modify the Script:**

   Update your script to save both old and new versions of `requirements.txt` into `requirements_history` folder with a timestamp.

```python
import subprocess
import re
import os
import shutil
from datetime import datetime

def get_stable_version(package_name):
    """Get the latest stable version of a package using pip."""
    result = subprocess.run(['pip', 'search', package_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error searching for package {package_name}")
        return None

    match = re.search(f'^{package_name} \((.*?)\)', result.stdout, re.MULTILINE)
    if match:
        return match.group(1)
    else:
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
            package_name = line.split('==')[0]
            stable_version = get_stable_version(package_name)
            if stable_version:
                updated_lines.append(f'{package_name}=={stable_version}\n')
            else:
                updated_lines.append(line)

        with open(new_file_path, 'w') as file:
            file.writelines(updated_lines)

        shutil.copy(new_file_path, requirements_file)
        print("requirements.txt has been updated with stable versions.")
        print(f"Old requirements saved to {old_file_path}")
        print(f"New requirements saved to {new_file_path}")
    except FileNotFoundError:
        print("requirements.txt not found.")

if __name__ == "__main__":
    update_requirements_file()
```

### Step-by-Step Implementation

1. **Create the Folder:**

   Make sure there is a `requirements_history` folder in your project directory.

   ```sh
   mkdir requirements_history
   ```

2. **Make the Script Executable:**

   Ensure the script is executable.

   ```sh
   chmod +x update_requirements.py
   ```

3. **Create the Git Hook:**

   Navigate to the `.git/hooks` directory in your repository:

   ```sh
   cd /path/to/your/repo/.git/hooks
   cd /home/siuadmin/temp/major/Research_docs/installations/pip_install_from_req.py #from my laptop path
   ```

   Create or edit the `post-merge` hook:

   ```sh
   nano post-merge
   ```

   Add the following lines to the `post-merge` file:

   ```sh
   #!/bin/sh
   python3 /path/to/your/repo/update_requirements.py
   ```

   Replace `/path/to/your/repo` with the actual path to your repository.

4. **Make the Git Hook Executable:**

   ```sh
   chmod +x post-merge
   ```

### Summary

Now, every time you run `git pull`, the `post-merge` hook will execute your `update_requirements.py` script, which will:

- Save the current `requirements.txt` as `requirements_<timestamp>_old.txt` in the `requirements_history` folder.
- Create a new `requirements_<timestamp>_new.txt` in the `requirements_history` folder with the updated stable versions.
- Replace the current `requirements.txt` with the new updated file.

This will help you keep track of the old and new versions of your `requirements.txt` file.
