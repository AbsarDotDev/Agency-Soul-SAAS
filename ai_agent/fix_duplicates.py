"""
Fix the duplicate function and other issues in the visualization agent
"""
import os
import re

# Path to the file
file_path = '/Applications/XAMPP/xamppfiles/htdocs/nomessos/ai_agent/app/visualizations/langgraph/visualization_agent.py'

# 1. Read the file
with open(file_path, 'r') as f:
    content = f.read()

# 2. Find the duplicate _create_default_chart_structure method
pattern = r'def _create_default_chart_structure\(self,.*?\):.*?return {.*?"chart_type":\s*"bar".*?}.*?def _create_default_chart_structure\(self,'
duplicate_method = re.search(pattern, content, re.DOTALL)

if duplicate_method:
    # Calculate the start and end positions of the first occurrence
    start_pos = duplicate_method.start()
    # Find the end of the first method definition by searching for the start of the second one
    second_def_pos = content.find('def _create_default_chart_structure(self,', start_pos + 10)
    
    if second_def_pos > 0:
        # Replace the entire first method with a comment
        new_content = content[:start_pos] + "    # Duplicate method removed - using the implementation below\n" + content[second_def_pos:]
        
        # Write back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print("Successfully removed duplicate method!")
    else:
        print("Could not find the end of the first method")
else:
    print("No duplicate method found - perhaps already fixed")
