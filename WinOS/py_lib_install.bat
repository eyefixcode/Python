@echo off
:install_libraries
cls

echo Welcome to the Python library installation tool!
echo Please enter the name of the library you want to install (or 'q' to quit):

set /p library_name=

if /i "%library_name%"=="q" goto :eof

pip install %library_name%

if %errorlevel% equ 0 (
    echo Library '%library_name%' installed successfully!
) else (
    echo Error: Failed to install library '%library_name%'.
    echo Please check for errors or try installing manually.
)

pause
goto install_libraries
