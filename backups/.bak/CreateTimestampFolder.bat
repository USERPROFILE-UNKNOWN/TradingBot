@echo off
setlocal enableextensions

:: Get current date and time components
for /f "tokens=2 delims==." %%A in ('"wmic os get localdatetime /value"') do set datetime=%%A

:: Format to [YYYY.MM.DD_HH.MM.SS]
set "YYYY=%datetime:~0,4%"
set "MM=%datetime:~4,2%"
set "DD=%datetime:~6,2%"
set "HH=%datetime:~8,2%"
set "Min=%datetime:~10,2%"
set "SS=%datetime:~12,2%"
set "foldername=[%YYYY%.%MM%.%DD%_%HH%.%Min%.%SS%]"

:: Create folder in the same directory as the .bat file
pushd "%~dp0"
mkdir "%foldername%"
echo Folder created: %foldername%
popd
endlocal
