@echo off

chcp 65001


pushd %~dp0

SET "SCRIPT_DIR=%CD:\=/%"

SET "EXPORT_DIR=%SCRIPT_DIR%/Output"
SET "EXPORT_OP_DIR=%SCRIPT_DIR%/Output/Optimized"

SET "F5_TTS_PROJECT_DIR=%SCRIPT_DIR%/F5-TTS"
SET "DOWNLOAD_DIR=%SCRIPT_DIR%/Downloads"
SET "INFER_PY_FILE=%SCRIPT_DIR%/Export_ONNX/F5_TTS/F5-TTS-ONNX-Inference.py"
SET "EXPORT_PY_FILE=%SCRIPT_DIR%/Export_ONNX/F5_TTS/Export_F5.py"
SET "OPTIMIZE_PY_FILE=%SCRIPT_DIR%/Export_ONNX/F5_TTS/Optimize_ONNX_DML.py"

SET "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%/F5-TTS"
SET "VOCOS_DIR=%DOWNLOAD_DIR%/vocos-mel-24khz"


@rem This for export_F5.py

SET "F5_project_path=%F5_TTS_PROJECT_DIR%"


@echo Work DIR: %SCRIPT_DIR%


if NOT EXIST "%DOWNLOAD_DIR%" (
    mkdir "%DOWNLOAD_DIR%"
)

if NOT EXIST "%EXPORT_DIR%" (
    mkdir "%EXPORT_DIR%"
)

if NOT EXIST "%EXPORT_OP_DIR%" (
    mkdir "%EXPORT_OP_DIR%"
)

if NOT EXIST "%F5_TTS_PROJECT_DIR%" (
    @echo Clone F5-TTS project ...
    @echo.
    git clone https://github.com/SWivid/F5-TTS.git || exit

    cd "%F5_TTS_PROJECT_DIR%"
    pip install f5-tts
)
  

if NOT EXIST "%F5_TTS_MODEL_DIR%" (
    @echo Clone F5-TTS models from huggingface ...
    @echo.
    git clone https://huggingface.co/SWivid/F5-TTS "%F5_TTS_MODEL_DIR%"
)

if NOT EXIST "%VOCOS_DIR%" (
    @echo Clone vocos models from huggingface ...
    @echo.
    git clone https://huggingface.co/charactr/vocos-mel-24khz "%VOCOS_DIR%"
)

pushd %SCRIPT_DIR%

@echo.
pip show onnxruntime-tools

if "%errorlevel%"=="0" (
    @echo skip install onnxruntime ...
) else (
    pip install onnxruntime-tools==1.7.0
    pip install onnxslim onnxconverter_common --upgrade
    @rem fix numpy version
    pip install numba --upgrade 
)

@REM Use bellow, if you want to use directml
@REM pip show onnxruntime-directml
@REM if "%errorlevel%"=="0" (
@REM     @echo skip install onnxruntime-directml ...
@REM ) else (
@REM     @echo.
@REM     @echo start install onnxruntime-directml ...
@REM     @echo.
@REM     pip uninstall onnxruntime onnxruntime-gpu -y
@REM     python -m pip install --force-reinstall onnxruntime-directml
@REM )


@rem change code for windows
@echo modify code ...
@echo.
@echo modify %EXPORT_PY_FILE%



call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5TTS_v1_Base" "%F5_TTS_MODEL_DIR%/F5TTS_v1_Base"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/vocos-mel-24khz" "%VOCOS_DIR%"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5_ONNX/" "%EXPORT_DIR%/"

@echo modify %OPTIMIZE_PY_FILE%
call:ReplaceString "%OPTIMIZE_PY_FILE%" "/home/DakeQQ/Downloads/F5_ONNX" "%EXPORT_DIR:/=\\%"
call:ReplaceString "%OPTIMIZE_PY_FILE%" "/home/DakeQQ/Downloads/F5_Optimized" "%EXPORT_OP_DIR:/=\\%"


@echo modify %INFER_PY_FILE%
call:ReplaceString "%INFER_PY_FILE%" "F5_Preprocess.ort" "F5_Preprocess.onnx"
call:ReplaceString "%INFER_PY_FILE%" "F5_Decode.ort" "F5_Decode.onnx"
call:ReplaceString "%INFER_PY_FILE%" "/home/DakeQQ/Downloads/F5TTS_v1_Base" "%F5_TTS_MODEL_DIR%/F5TTS_v1_Base"
call:ReplaceString "%INFER_PY_FILE%" "/home/DakeQQ/Downloads/F5_Optimized/" "%EXPORT_DIR%/"

@REM ---------------------- Start Python Execution--------------------------

cd "%SCRIPT_DIR%/Export_ONNX/F5_TTS"

python ./Export_F5.py || goto OnError

@echo.
@echo Model exported at %EXPORT_DIR%
@echo.

set /p OP="Continue Optimize Model ? y/n[n]"

if /i "%OP%" == "y" (
    @echo All Model Exported, Wait Optimize ...

    python ./Optimize_ONNX_DML.py --model "F5_Transformer.onnx"

    @echo.
    @echo All Done !!
)

@REM If you want to inference, use bellow:
@REM python "%INFER_PY_FILE%"


@rem restore code ? if you want
@rem git checkout -- Export_F5.py
@rem git checkout -- Optimize_ONNX.py
exit

:OnError
echo.
echo Process failed !!!!
echo.
exit
GOTO:EOF


:ReplaceString
set ff=%~1
set old_str=%~2
set new_str=%~3
powershell -Command "(gc -Encoding UTF8 %ff%) -replace '%old_str%', '%new_str%' | Set-Content -Encoding UTF8 %ff%"
GOTO:EOF
