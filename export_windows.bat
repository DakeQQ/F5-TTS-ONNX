@echo off

chcp 65001


pushd %~dp0

SET "SCRIPT_DIR=%CD:\=/%"

SET "EXPORT_DIR=%SCRIPT_DIR%/Output"
SET "EXPORT_OP_DIR=%SCRIPT_DIR%/Output/Optimized"

SET "F5_TTS_PROJECT_DIR=%SCRIPT_DIR%/F5-TTS"
SET "DOWNLOAD_DIR=%SCRIPT_DIR%/Downloads"
SET "INFER_PY_FILE=%SCRIPT_DIR%/F5-TTS-ONNX-Inference.py"
SET "EXPORT_PY_FILE=%SCRIPT_DIR%/Export_ONNX/F5_TTS/Export_F5.py"
SET "OPTIMIZE_PY_FILE=%SCRIPT_DIR%/Export_ONNX/F5_TTS/Optimize_ONNX.py"

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
    git clone https://github.com/SWivid/F5-TTS.git
    
    cd "%F5_TTS_PROJECT_DIR%"
    pip install -e .
)

if NOT EXIST "%F5_TTS_MODEL_DIR%" (
    git clone https://huggingface.co/SWivid/F5-TTS "%F5_TTS_MODEL_DIR%"
)

if NOT EXIST "%VOCOS_DIR%" (
    git clone https://huggingface.co/charactr/vocos-mel-24khz "%VOCOS_DIR%"
)

pushd %SCRIPT_DIR%

@echo.
pip install onnxruntime-tools==1.7.0
pip install onnxslim onnxconverter_common --upgrade

@REM Use bellow, if you want to use directml
@rem pip uninstall onnxruntime onnxruntime-gpu -y
@rem python -m pip install --force-reinstall onnxruntime-directml

@rem change code for windows
@echo modify code ...
@echo.
@echo modify %EXPORT_PY_FILE%

call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5-TTS-main/src/f5_tts/infer/examples/basic/generated.wav" "%EXPORT_DIR%/infer_result.wav"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5-TTS-main" "%F5_TTS_PROJECT_DIR%"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5-TTS_Emilia-ZH-EN/F5TTS_Base" "%F5_TTS_MODEL_DIR%/F5TTS_Base"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/vocos-mel-24khz" "%VOCOS_DIR%"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5_ONNX/" "%EXPORT_DIR%/"

@echo modify %OPTIMIZE_PY_FILE%
call:ReplaceString "%OPTIMIZE_PY_FILE%" "/home/DakeQQ/Downloads/F5_ONNX" "%EXPORT_DIR:/=\\%"
call:ReplaceString "%OPTIMIZE_PY_FILE%" "/home/DakeQQ/Downloads/F5_Optimized" "%EXPORT_OP_DIR:/=\\%"


@echo modify %INFER_PY_FILE%
call:ReplaceString "%INFER_PY_FILE%" "F5_Preprocess.ort" "F5_Preprocess.onnx"
call:ReplaceString "%INFER_PY_FILE%" "F5_Decode.ort" "F5_Decode.onnx"
call:ReplaceString "%INFER_PY_FILE%" "/home/DakeQQ/Downloads/F5-TTS-main/src/f5_tts/infer/examples/basic/generated.wav" "%EXPORT_DIR%/infer_result.wav"
call:ReplaceString "%INFER_PY_FILE%" "/home/DakeQQ/Downloads/F5-TTS-main" "%F5_TTS_PROJECT_DIR%"
call:ReplaceString "%INFER_PY_FILE%" "/home/DakeQQ/Downloads/F5_Optimized/" "%EXPORT_DIR%/"

@REM ---------------------- Start Python Execution--------------------------

cd "%SCRIPT_DIR%/Export_ONNX/F5_TTS"

@rem python ./Export_F5.py || goto OnError

set /p OP="Continue Optimize Model ? y/n[n]"

if /i "%OP%" == "y" (
    @echo All Model Exported, Wait Optimize ...

    @rem python ./Optimize_ONNX_DML.py --model "F5_Preprocess.onnx"
    @rem python ./Optimize_ONNX_DML.py --model "F5_Transformer.onnx"
    @rem python ./Optimize_ONNX.py --model "F5_Decode.onnx"

    @echo.
    @echo All Done !!
)

@REM If you want to inference, use bellow:
python "%SCRIPT_DIR%/F5-TTS-ONNX-Inference.py"


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
