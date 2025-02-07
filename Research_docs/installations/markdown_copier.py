import os
import shutil

def copy_markdown_files(src_dirs, dest_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over each source directory
    for src_dir in src_dirs:
        # Check if source directory exists
        if not os.path.exists(src_dir):
            raise ValueError(f"Source directory '{src_dir}' does not exist.")
        
        # Walk through the source directory
        for root, dirs, files in os.walk(src_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.md'):
                    # Construct full file paths
                    src_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, src_dir)
                    dest_file_dir = os.path.join(dest_dir, os.path.basename(src_dir), relative_path)
                    dest_file_path = os.path.join(dest_file_dir, file)
                    
                    # Create destination subdirectory if it doesn't exist
                    os.makedirs(dest_file_dir, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(src_file_path, dest_file_path)
                    print(f"Copied: {src_file_path} to {dest_file_path}")


# Example usage
src_directories = [
    '/home/siuadmin/temp/Windows_application',
    "/home/siuadmin",
]
dest_directory = '/mnt/c/Users/devil/Documents/test/test/imp'
copy_markdown_files(src_directories, dest_directory)
