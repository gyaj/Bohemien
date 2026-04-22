@echo off
setlocal
:: Run PyInstaller from the PARENT of Bohemien_Motor_Designer\ (one level up)
cd /d "%~dp0.."
echo Building from: %CD%

echo Installing dependencies...
pip install pyinstaller numpy scipy matplotlib --quiet
if errorlevel 1 (echo pip failed & pause & exit /b 1)

echo Cleaning previous build...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo Building exe (may take 3-8 minutes)...
pyinstaller Bohemien_Motor_Designer\Bohemien_Motor_Designer.spec --clean --noconfirm
if errorlevel 1 (
    echo BUILD FAILED. Open Bohemien_Motor_Designer.spec, set console=True, rebuild,
    echo then run dist\Bohemien_Motor_Designer.exe from CMD to see the error.
    pause & exit /b 1
)

echo Done!
if exist dist\Bohemien_Motor_Designer.exe (
    echo EXE: %CD%\dist\Bohemien_Motor_Designer.exe
    for %%F in (dist\Bohemien_Motor_Designer.exe) do echo Size: %%~zF bytes
)
pause
