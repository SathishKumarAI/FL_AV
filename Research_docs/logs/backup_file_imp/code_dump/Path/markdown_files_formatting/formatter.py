import os
import re

def create_markdown_with_heading(input_file_path):
    with open(input_file_path, 'r') as input_file:
        file_content = input_file.read()

    # Find all blocks of content starting with ** followed by an integer
    pattern = re.compile(r'\*\*(\d+.*?)\*\*')
    matches = pattern.finditer(file_content)

    # Create output file path by appending "_updated.md" to the input file name
    output_file_path = os.path.splitext(input_file_path)[0] + "_updated.md"

    with open(output_file_path, 'w') as output_file:
        cursor = 0
        for match in matches:
            start_idx = match.start()
            end_idx = match.end()

            # Write the content before the current block
            output_file.write(file_content[cursor:start_idx])

            # Write the heading and the block content
            block_content = match.group(1).strip()
            heading = f'### {block_content}\n'
            output_file.write(heading)

            cursor = end_idx

        # Write any remaining content after the last block
        output_file.write(file_content[cursor:])

if __name__ == "__main__":
    input_file_path = "/mnt/c/Users/devil/Documents/test/test/imp/siuadmin/data engineering/SQL questions.md"
    create_markdown_with_heading(input_file_path)
