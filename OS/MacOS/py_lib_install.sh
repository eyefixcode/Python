#!/bin/bash

while true; do
  # Open Terminal and keep it focused
  osascript -e 'tell application "Terminal" to activate'

  # Display instructions in the Terminal window
  echo "Welcome to the Python library installation tool!"
  echo "Please enter the name of the library you want to install (or 'q' to quit):"

  # Wait for user input
  read library_name

  if [[ $library_name == "q" ]]; then
    break  # Exit the loop if 'q' is entered
  fi

  # Attempt to install the library using pip3
  pip3 install $library_name

  # Check the installation status and provide feedback
  if [[ $? -eq 0 ]]; then
    echo "Library '$library_name' installed successfully!"
  else
    echo "Error: Failed to install library '$library_name'."
    echo "Please check for errors or try installing manually."
  fi

  # Close the Terminal window (attempting different strategies)
  osascript -e 'tell application "Terminal" to close (every window whose frontmost is true)'
  osascript -e 'tell application "Terminal" to close window whose (its visible processes as string) contains "Welcome to the Python library installation tool!"'
done