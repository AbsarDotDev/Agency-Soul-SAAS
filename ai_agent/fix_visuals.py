import sys
import os

# Open the file
filepath = "/Applications/XAMPP/xamppfiles/htdocs/nomessos/ai_agent/app/visualizations/langgraph/visualization_agent.py"
with open(filepath, 'r') as f:
    lines = f.readlines()

# Find the _create_default_sql_query method
found_default_sql_query = False
for i, line in enumerate(lines):
    if "_create_default_sql_query" in line and "def" in line:
        found_default_sql_query = True
        print(f"Found _create_default_sql_query at line {i+1}")
        
# Update is_relevant logic
for i, line in enumerate(lines):
    if "if not parsed_question.get('is_relevant', False):" in line and "logger.error" in lines[i+1]:
        lines[i+1] = lines[i+1].replace("logger.error", "logger.warning")
        print(f"Updated is_relevant error to warning at line {i+2}")
        
    # Fix the SQL generation logic to use default SQL
    elif "# Make sure we have a proper SQL query" in line and i+10 < len(lines):
        for j in range(i, i+20):
            if "if not sql_query or len(sql_query) < 10 or" in lines[j]:
                next_line = lines[j+1]
                # Ensure we use the default SQL when possible
                if "logger.error" in next_line and "default_sql" not in next_line:
                    lines[j+1] = next_line.replace("logger.error", "logger.warning")
                    print(f"Updated SQL generation logic at line {j+2}")

# Write the modified content back
with open(filepath, 'w') as f:
    f.writelines(lines)

print("Done patching visualization file!")
