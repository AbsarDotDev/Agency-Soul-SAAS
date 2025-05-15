import re

# Path to the file
file_path = '/Applications/XAMPP/xamppfiles/htdocs/nomessos/ai_agent/app/visualizations/langgraph/visualization_agent.py'

with open(file_path, 'r') as file:
    lines = file.readlines()

fixed_lines = []
skip_line = False
in_problematic_method = False

# Process the file line by line
for i, line in enumerate(lines):
    # Skip problematic indented blocks
    if in_problematic_method and line.strip() and line.startswith(' ' * 8):
        if any(pattern in line for pattern in ["labels", "values_column", "labels_column"]):
            skip_line = True
        else:
            skip_line = False
    else:
        skip_line = False
    
    # Check if we're starting a problematic method
    if "_create_default_chart_structure" in line and "def" in line:
        # Skip the first occurrence but keep others
        if in_problematic_method:
            in_problematic_method = False
        else:
            in_problematic_method = True
    
    # Add the line if it's not being skipped
    if not skip_line:
        fixed_lines.append(line)

# Write the fixed content back
with open(file_path, 'w') as file:
    file.writelines(fixed_lines)

# Now check if there are any errors in the fixed file
import subprocess
result = subprocess.run(['python', '-m', 'py_compile', file_path], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("Successfully fixed syntax errors!")
else:
    print("Still have syntax errors:")
    print(result.stderr)
