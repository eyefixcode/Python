#!/bin/bash

# Navigate to the desktop
cd ~/Desktop

# Prompt the user for the waiter .sh file name
read -p "Enter the name of the waiter .sh file (excluding .sh): " filename

# Append .sh to the filename
filename="${filename}.sh"

# Make the waiter .sh file executable
chmod +x "${filename}"

# Optionally, launch the waiter .sh file (uncomment if desired)
# ./"${filename}"