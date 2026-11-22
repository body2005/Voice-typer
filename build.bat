@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo        Building VoiceTyper with PyInstaller
echo ========================================================

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "E:\voice2text\.venv\Scripts\activate.bat" (
    call E:\voice2text\.venv\Scripts\activate.bat
)

REM Terminate any running VoiceTyper instances to prevent file lock errors
taskkill /F /IM VoiceTyper.exe /IM VoiceTyper-Standalone.exe 2>nul

REM Ensure assets exist
if not exist "voicetyper\assets\icon.ico" (
    echo [ERROR] Assets not found in voicetyper\assets.
    exit /b 1
)

echo [1/2] Building Onedir variant (Recommended: fast startup, low false positives)...
pyinstaller --noconfirm --clean --noconsole --onedir --noupx ^
    --name "VoiceTyper" ^
    --icon "voicetyper\assets\icon.ico" ^
    --version-file "version_info.txt" ^
    --add-data "voicetyper\assets;assets" ^
    --add-data "voicetyper\bin\whisper-vulkan;bin/whisper-vulkan" ^
    --exclude-module "PyQt5" ^
    --exclude-module "matplotlib" ^
    --exclude-module "IPython" ^
    --hidden-import "PyQt6" ^
    --hidden-import "keyring.backends.Windows" ^
    --hidden-import "scipy.signal" ^
    --hidden-import "hf_transfer" ^
    --collect-all "faster_whisper" ^
    --collect-all "ctranslate2" ^
    --collect-all "huggingface_hub" ^
    --collect-all "tokenizers" ^
    --collect-all "onnxruntime" ^
    --collect-all "tqdm" ^
    --collect-all "certifi" ^
    --collect-all "requests" ^
    main.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Onedir build failed.
    exit /b %ERRORLEVEL%
)

if exist "dist\VoiceTyper" (
    if exist "voicetyper\bin\whisper-vulkan" (
        xcopy /E /I /Y "voicetyper\bin\whisper-vulkan" "dist\VoiceTyper\bin\whisper-vulkan" >nul 2>nul
    )
)

echo [2/2] Building Onefile standalone variant (Optional)...
taskkill /F /IM VoiceTyper.exe /IM VoiceTyper-Standalone.exe 2>nul
del /f /q dist\VoiceTyper-Standalone.exe 2>nul
pyinstaller --noconfirm --clean --noconsole --onefile --noupx ^
    --name "VoiceTyper-Standalone" ^
    --icon "voicetyper\assets\icon.ico" ^
    --version-file "version_info.txt" ^
    --add-data "voicetyper\assets;assets" ^
    --add-data "voicetyper\bin\whisper-vulkan;bin/whisper-vulkan" ^
    --exclude-module "PyQt5" ^
    --exclude-module "matplotlib" ^
    --exclude-module "IPython" ^
    --hidden-import "PyQt6" ^
    --hidden-import "keyring.backends.Windows" ^
    --hidden-import "scipy.signal" ^
    --hidden-import "hf_transfer" ^
    --collect-all "faster_whisper" ^
    --collect-all "ctranslate2" ^
    --collect-all "huggingface_hub" ^
    --collect-all "tokenizers" ^
    --collect-all "onnxruntime" ^
    --collect-all "tqdm" ^
    --collect-all "certifi" ^
    --collect-all "requests" ^
    main.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Onefile build failed.
    exit /b %ERRORLEVEL%
)

echo [3/3] Signing binaries with Authenticode Digital Certificate...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Select-Object -First 1; if (-not $cert) { $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=VoiceTyper Open Source, O=VoiceTyper, C=US' -CertStoreLocation Cert:\CurrentUser\My }; if (Test-Path 'dist\VoiceTyper-Standalone.exe') { Set-AuthenticodeSignature -Certificate $cert -FilePath 'dist\VoiceTyper-Standalone.exe' | Out-Null }; if (Test-Path 'dist\VoiceTyper\VoiceTyper.exe') { Set-AuthenticodeSignature -Certificate $cert -FilePath 'dist\VoiceTyper\VoiceTyper.exe' | Out-Null }; Write-Host '[SUCCESS] Authenticode signature applied successfully.'"

echo ========================================================
echo Build complete!
echo Output binaries located in dist/VoiceTyper and dist/VoiceTyper-Standalone.exe
echo ========================================================
