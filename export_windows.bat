@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

pushd "%~dp0" || exit /b 1
set "SCRIPT_DIR=%CD%"
set "F5_DIR=%SCRIPT_DIR%\Export_ONNX\F5_TTS"
set "EXPORT_PY_FILE=%F5_DIR%\Export_F5.py"
set "OPTIMIZE_PY_FILE=%F5_DIR%\Optimize_ONNX_DML.py"
set "INFER_PY_FILE=%F5_DIR%\F5-TTS-ONNX-Inference.py"
set "REQUIREMENTS_FILE=%F5_DIR%\requirements.txt"
set "EXPORT_DIR=%SCRIPT_DIR%\Output"
set "EXPORT_OP_DIR=%EXPORT_DIR%\Optimized"
set "DOWNLOAD_DIR=%USERPROFILE%\Downloads"
set "VOCOS_DIR=%DOWNLOAD_DIR%\vocos-mel-24khz"
set "VOCOS_CONFIG=%VOCOS_DIR%\config.yaml"
set "VOCOS_WEIGHTS=%VOCOS_DIR%\pytorch_model.bin"
set "OPTIMIZE_MODE=prompt"
set "MODEL_SERIES=v1"

goto :ParseArgs

:ParseArgs
if "%~1"=="" goto :ArgsReady
if /i "%~1"=="--optimize" (
    set "OPTIMIZE_MODE=yes"
    shift
    goto :ParseArgs
)
if /i "%~1"=="--no-optimize" (
    set "OPTIMIZE_MODE=no"
    shift
    goto :ParseArgs
)
if /i "%~1"=="--model-series" (
    if "%~2"=="" goto :InvalidArgument
    set "MODEL_SERIES=%~2"
    shift
    shift
    goto :ParseArgs
)
if /i "%~1"=="--help" goto :Usage
if /i "%~1"=="/?" goto :Usage
goto :InvalidArgument

:ArgsReady
if /i not "%MODEL_SERIES%"=="v0" if /i not "%MODEL_SERIES%"=="v1" goto :InvalidArgument
call :ResolveModelPaths

echo Working directory: %SCRIPT_DIR%
echo Model series: %MODEL_SERIES%
echo.

where python >nul 2>&1 || (
    echo [ERROR] Python was not found on PATH.
    goto :Failed
)
python -c "import sys; assert sys.version_info >= (3, 10), sys.version" >nul 2>&1 || (
    echo [ERROR] Python 3.10 or newer is required.
    goto :Failed
)
python -m pip --version >nul 2>&1 || (
    echo [ERROR] pip is unavailable for the selected Python interpreter.
    goto :Failed
)

echo [1/4] Checking Python dependencies...
python -c "import f5_tts, huggingface_hub, numpy, omegaconf, onnx, onnxconverter_common, onnxruntime, onnxslim, pydub, pypinyin, rjieba, safetensors, soundfile, torch, torchaudio, vocos, x_transformers, yaml" >nul 2>&1
if errorlevel 1 (
    echo Installing missing export dependencies...
    python -m pip install -r "%REQUIREMENTS_FILE%" || goto :Failed
    python -c "import f5_tts, huggingface_hub, numpy, omegaconf, onnx, onnxconverter_common, onnxruntime, onnxslim, pydub, pypinyin, rjieba, safetensors, soundfile, torch, torchaudio, vocos, x_transformers, yaml" >nul 2>&1 || (
        echo [ERROR] One or more Python dependencies still cannot be imported.
        goto :Failed
    )
)

if not exist "%DOWNLOAD_DIR%" mkdir "%DOWNLOAD_DIR%" || goto :Failed
if not exist "%EXPORT_DIR%" mkdir "%EXPORT_DIR%" || goto :Failed
if not exist "%EXPORT_OP_DIR%" mkdir "%EXPORT_OP_DIR%" || goto :Failed

echo [2/4] Checking model files...
set "DOWNLOAD_F5="
if not exist "%F5_TTS_CHECKPOINT%" set "DOWNLOAD_F5=1"
if not exist "%F5_TTS_VOCAB%" set "DOWNLOAD_F5=1"
if defined DOWNLOAD_F5 (
    echo Downloading F5-TTS %MODEL_SERIES% assets...
    if /i "%MODEL_SERIES%"=="v0" (
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SWivid/F5-TTS', local_dir=r'%DOWNLOAD_DIR%', allow_patterns=['F5TTS_v0_Base/*', 'F5TTS_Base/*'])" || goto :Failed
    ) else (
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SWivid/F5-TTS', local_dir=r'%DOWNLOAD_DIR%', allow_patterns=['F5TTS_v1_Base/*'])" || goto :Failed
    )
    call :ResolveModelPaths
)

set "DOWNLOAD_VOCOS="
if not exist "%VOCOS_CONFIG%" set "DOWNLOAD_VOCOS=1"
if not exist "%VOCOS_WEIGHTS%" set "DOWNLOAD_VOCOS=1"
if defined DOWNLOAD_VOCOS (
    echo Downloading Vocos assets...
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='charactr/vocos-mel-24khz', local_dir=r'%VOCOS_DIR%')" || goto :Failed
)

if not exist "%F5_TTS_CHECKPOINT%" (
    echo [ERROR] F5-TTS checkpoint was not found at "%F5_TTS_CHECKPOINT%".
    goto :Failed
)
if not exist "%F5_TTS_VOCAB%" (
    echo [ERROR] F5-TTS vocabulary was not found at "%F5_TTS_VOCAB%".
    goto :Failed
)
if not exist "%VOCOS_CONFIG%" (
    echo [ERROR] Vocos config was not found at "%VOCOS_CONFIG%".
    goto :Failed
)
if not exist "%VOCOS_WEIGHTS%" (
    echo [ERROR] Vocos weights were not found at "%VOCOS_WEIGHTS%".
    goto :Failed
)

echo [3/4] Exporting F5-TTS ONNX models...
python "%EXPORT_PY_FILE%" ^
    --model_series "%MODEL_SERIES%" ^
    --f5safetensor_path "%F5_TTS_CHECKPOINT%" ^
    --vocab_path "%F5_TTS_VOCAB%" ^
    --vocosmodel_dir "%VOCOS_DIR%" ^
    --preprocessmodel_path "%EXPORT_DIR%\F5_Preprocess.onnx" ^
    --transformermodel_path "%EXPORT_DIR%\F5_Transformer.onnx" ^
    --decodermodel_path "%EXPORT_DIR%\F5_Decode.onnx" ^
    --metadatamodel_path "%EXPORT_DIR%\F5_Metadata.onnx" || goto :Failed
for %%F in (F5_Preprocess.onnx F5_Transformer.onnx F5_Decode.onnx F5_Metadata.onnx) do (
    if not exist "%EXPORT_DIR%\%%F" (
        echo [ERROR] Export completed without creating "%EXPORT_DIR%\%%F".
        goto :Failed
    )
)
echo Exported models: %EXPORT_DIR%
echo.

if /i "%OPTIMIZE_MODE%"=="prompt" (
    choice /c YN /n /m "Optimize all models for DirectML? [Y/N]: "
    if errorlevel 3 goto :Failed
    if errorlevel 2 (
        set "OPTIMIZE_MODE=no"
    ) else if errorlevel 1 (
        set "OPTIMIZE_MODE=yes"
    ) else (
        goto :Failed
    )
)

if /i "%OPTIMIZE_MODE%"=="yes" (
    echo [4/4] Optimizing models for DirectML...
    python "%OPTIMIZE_PY_FILE%" --source-folder "%EXPORT_DIR%" --output-folder "%EXPORT_OP_DIR%" || goto :Failed
    for %%F in (F5_Preprocess.onnx F5_Transformer.onnx F5_Decode.onnx F5_Metadata.onnx) do (
        if not exist "%EXPORT_OP_DIR%\%%F" (
            echo [ERROR] Optimization completed without creating "%EXPORT_OP_DIR%\%%F".
            goto :Failed
        )
    )
    echo Optimized models: %EXPORT_OP_DIR%
) else (
    echo [4/4] DirectML optimization skipped.
)

echo.
echo Completed successfully.
echo To run inference:
echo   python "%INFER_PY_FILE%" --metadata_path "%EXPORT_DIR%\F5_Metadata.onnx" --vocab_path "%F5_TTS_VOCAB%"
goto :Success

:ResolveModelPaths
if /i "%MODEL_SERIES%"=="v0" (
    set "F5_TTS_MODEL_DIR="
    if exist "%DOWNLOAD_DIR%\F5TTS_v0_Base\model_1200000.safetensors" if exist "%DOWNLOAD_DIR%\F5TTS_v0_Base\vocab.txt" set "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%\F5TTS_v0_Base"
    if not defined F5_TTS_MODEL_DIR if exist "%DOWNLOAD_DIR%\F5TTS_Base\model_1200000.safetensors" if exist "%DOWNLOAD_DIR%\F5TTS_Base\vocab.txt" set "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%\F5TTS_Base"
    if not defined F5_TTS_MODEL_DIR if exist "%DOWNLOAD_DIR%\F5TTS_v1_Base\model_1200000.safetensors" if exist "%DOWNLOAD_DIR%\F5TTS_v1_Base\vocab.txt" set "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%\F5TTS_v1_Base"
    if not defined F5_TTS_MODEL_DIR set "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%\F5TTS_v0_Base"
    set "F5_TTS_CHECKPOINT=%F5_TTS_MODEL_DIR%\model_1200000.safetensors"
) else (
    set "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%\F5TTS_v1_Base"
    set "F5_TTS_CHECKPOINT=%F5_TTS_MODEL_DIR%\model_1250000.safetensors"
)
set "F5_TTS_VOCAB=%F5_TTS_MODEL_DIR%\vocab.txt"
exit /b 0

:InvalidArgument
echo [ERROR] Invalid command line.
echo Usage: %~nx0 [--optimize ^| --no-optimize] [--model-series v0 ^| v1]
goto :Failed

:Usage
echo Usage: %~nx0 [--optimize ^| --no-optimize] [--model-series v0 ^| v1]
echo.
echo   --optimize          Export and optimize for DirectML without prompting.
echo   --no-optimize       Export only without prompting.
echo   --model-series v0   Export F5TTS_Base.
echo   --model-series v1   Export F5TTS_v1_Base. This is the default.
goto :Success

:Failed
echo.
echo Process failed.
popd
endlocal
exit /b 1

:Success
popd
endlocal
exit /b 0
