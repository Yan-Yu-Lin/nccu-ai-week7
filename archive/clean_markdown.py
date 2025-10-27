#!/usr/bin/env python3
"""
Script to clean markdown files by removing bash, shell, and JavaScript code blocks.
Preserves Python code blocks and all markdown documentation text.
"""

import re
import sys
from pathlib import Path

def clean_markdown(content):
    """
    Remove bash/curl and JavaScript code blocks from markdown content.

    Args:
        content (str): The markdown content to clean

    Returns:
        str: Cleaned markdown content with unwanted code blocks removed
    """
    # Pattern to match code blocks with language identifiers
    # Matches: ```language, content, ```

    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is the start of a code block
        if line.strip().startswith('```'):
            # Extract the language identifier
            match = re.match(r'```(\w+)', line.strip())
            language = match.group(1) if match else ''

            # Check if this is a language we want to remove
            if language.lower() in ['bash', 'sh', 'shell', 'javascript', 'js', 'typescript', 'ts']:
                # Find the closing ```
                i += 1
                while i < len(lines):
                    if lines[i].strip().startswith('```'):
                        i += 1  # Skip the closing ```
                        break
                    i += 1
                continue
            # Check for curl commands in code blocks without language identifier
            elif language == '':
                # Peek ahead to see if it's a curl command
                j = i + 1
                is_curl = False
                while j < len(lines) and not lines[j].strip().startswith('```'):
                    if 'curl' in lines[j].lower():
                        is_curl = True
                        break
                    j += 1

                if is_curl:
                    # Skip this entire code block
                    i += 1
                    while i < len(lines):
                        if lines[i].strip().startswith('```'):
                            i += 1
                            break
                        i += 1
                    continue

        # Keep this line
        result.append(line)
        i += 1

    return '\n'.join(result)


def process_file(file_path, output_path=None):
    """
    Process a markdown file and save the cleaned version.

    Args:
        file_path (str): Path to the markdown file to clean
        output_path (str, optional): Path to save the cleaned file. If None, overwrites original.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return False

    print(f"Processing: {file_path}")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Clean the content
    cleaned_content = clean_markdown(content)

    # Save the cleaned content
    if output_path is None:
        output_path = file_path
    else:
        output_path = Path(output_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    print(f"Saved to: {output_path}")
    return True


def main():
    """Main function to process multiple markdown files."""

    # Define the directory and files to process
    base_dir = Path("/Users/linyanyu/Desktop/20-29-Development/21-Active-Projects/政大AI/colab_assignment/Week7/response API OpenAI GPT 5")

    files_to_process = [
        "function calling.md",
        "text generation.md",
        "Using GPT 5.md"
    ]

    if len(sys.argv) > 1:
        # Allow custom file paths from command line
        files_to_process = sys.argv[1:]
    else:
        # Use default directory
        files_to_process = [base_dir / f for f in files_to_process]

    print("=" * 60)
    print("Markdown Code Block Cleaner")
    print("Removing: bash, sh, shell, javascript, js, typescript, ts")
    print("Keeping: Python code blocks and markdown text")
    print("=" * 60)
    print()

    success_count = 0
    for file_path in files_to_process:
        if process_file(file_path):
            success_count += 1
        print()

    print("=" * 60)
    print(f"Successfully processed {success_count}/{len(files_to_process)} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
