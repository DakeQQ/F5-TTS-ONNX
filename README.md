
---

## F5-TTS-ONNX  
Run **F5-TTS** using ONNX Runtime for efficient and flexible text-to-speech processing.

### Updates  
- 2025/4/2: Reduce the use of 24 instances of `transpose()` and more than 100 instances of `unsqueeze()` operators. The number of nodes in `F5_Transformer.onnx` is approximately 1281, all of which can be placed on non-CPU providers. We recommend considering the use of I/O binding, a feature of ONNX Runtime, to achieve maximum performance.
- It currently support the latest [**SWivid/F5-TTS - v1.1.7**](https://github.com/SWivid/F5-TTS). Please `pip install f5-tts --upgrade` first. 
- 2025/3/05: The issue of silence output when using float16 has now been resolved. Please set `use_fp16_transformer = True  # (Export_F5.py, Line: 21)` before export.
- 2025/3/01: [endink](https://github.com/endink) Add a Windows one-key export script to facilitate the use of Windows integration users. The script will automatically install dependencies. Usage:
  ```
   conda create -n f5_tts_export python=3.10 -y
   
   conda activate f5_tts_export
   
   git clone https://github.com/DakeQQ/F5-TTS-ONNX.git
   
   cd F5-TTS-ONNX
   
   .\export_windows.bat
   ```


### Features  
1. **Windows OS + Intel/AMD/Nvidia GPU**:  
   - Easy solution using ONNX-DirectML for GPUs on Windows.  
   - Install ONNX Runtime DirectML:  
     ```bash
     pip install onnxruntime-directml --upgrade
     ```
2. **CPU Only**:
   - For users with 'CPU only' setups, including Intel or AMD, you can try using `['OpenVINOExecutionProvider']` and adding `provider_options` for a slight performance boost of around 5~20%.
   - ```python
     provider_options = [{
        'device_type' : 'CPU',
        'precision' : 'ACCURACY',
        'num_of_threads': MAX_THREADS,
        'num_streams': 1,
        'enable_opencl_throttling' : False,
        'enable_qdq_optimizer': False
     }]
     ```
   - Remember `pip uninstall onnxruntime-gpu` and `pip uninstall onnxruntime-directml` first. Next `pip install onnxruntime-openvino --upgrade`.
3. **Intel OpenVINO**:
   - If you are using a recent Intel chip, you can try `['OpenVINOExecutionProvider']` with provider_options `'device_type': 'XXX'`, where `XXX` can be one of the following options:  (No guarantee that it will work or function well)
     - `CPU`  
     - `GPU`  
     - `NPU`  
     - `AUTO:NPU,CPU`  
     - `AUTO:NPU,GPU`  
     - `AUTO:GPU,CPU`  
     - `AUTO:NPU,GPU,CPU`  
     - `HETERO:NPU,CPU`  
     - `HETERO:NPU,GPU`  
     - `HETERO:GPU,CPU`  
     - `HETERO:NPU,GPU,CPU`
   - Remember `pip uninstall onnxruntime-gpu` and `pip uninstall onnxruntime-directml` first. Next `pip install onnxruntime-openvino --upgrade`.
4. **Simple GUI Version**:  
   - Try the easy-to-use GUI version:  
      - [F5-TTS-ONNX GUI](https://github.com/patientx/F5-TTS-ONNX-gui)

5. **NVIDIA TensorRT Support**:  
   - For NVIDIA GPU optimization with TensorRT, visit:  
      - [F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster)

6. **Download**
   - [Link](https://huggingface.co/H5N1AIDS/F5-TTS-ONNX/tree/main)

### Learn More  
- Explore more related projects and resources:  
  [Project Overview](https://github.com/DakeQQ?tab=repositories)

---

## F5-TTS-ONNX  
通过 ONNX Runtime 运行 **F5-TTS**，实现高效灵活的文本转语音处理。

### 更新  
- 2025/4/02：减少使用24个`transpose()`和超过100个`unsqueeze()`运算符的实例。`F5_Transformer.onnx`中的节点数量约为1281，所有这些节点都可以放置在非CPU提供者上。我们建议使用ONNX Runtime的I/O绑定功能，以实现最大性能。
- 支持最新的 [**SWivid/F5-TTS - v1.1.7**](https://github.com/SWivid/F5-TTS)，请先`pip install f5-tts --upgrade`。
- 2025/3/05 使用 float16 时出现的静音输出问题现已解决。在导出之前，请设置 `use_fp16_transformer = True  # (Export_F5.py，第 21 行)`。
- 2025/3/01: [endink](https://github.com/endink) 添加一个 windows 一键导出脚本，方便广大 windows 集成用户使用，脚本会自动安装依赖。使用方法：
  ```
   conda create -n f5_tts_export python=3.10 -y
   
   conda activate f5_tts_export
   
   git clone https://github.com/DakeQQ/F5-TTS-ONNX.git
   
   cd F5-TTS-ONNX
   
   .\export_windows.bat
   ```

### 功能  
1. **Windows 操作系统 + Intel/AMD/Nvidia GPU**：  
   - 针对 GPU 的简单解决方案，通过 ONNX-DirectML 在 Windows 上运行。  
   - 安装 ONNX Runtime DirectML：  
     ```bash
     pip install onnxruntime-directml --upgrade
     ```
2. **仅CPU：**  
   - 对于仅使用CPU的用户（包括Intel或AMD），可以尝试使用`['OpenVINOExecutionProvider']`并添加`provider_options`，以获得大约5~20%的性能提升。
   - 示例代码：  
     ```python
     provider_options = [{
        'device_type': 'CPU',
        'precision': 'ACCURACY',
        'num_of_threads': MAX_THREADS,
        'num_streams': 1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer': False
     }]
     ```  
   - 请记得先执行 `pip uninstall onnxruntime-gpu` and `pip uninstall onnxruntime-directml`。 接下来 `pip install onnxruntime-openvino --upgrade`。 

3. **Intel OpenVINO：**  
   - 如果您使用的是近期的Intel芯片，可以尝试`['OpenVINOExecutionProvider']`，并设置`provider_options`中的`'device_type': 'XXX'`，其中`XXX`可以是以下选项之一： (不能保证其能够正常运行或运行良好)
     - `CPU`  
     - `GPU`  
     - `NPU`  
     - `AUTO:NPU,CPU`  
     - `AUTO:NPU,GPU`  
     - `AUTO:GPU,CPU`  
     - `AUTO:NPU,GPU,CPU`  
     - `HETERO:NPU,CPU`  
     - `HETERO:NPU,GPU`  
     - `HETERO:GPU,CPU`  
     - `HETERO:NPU,GPU,CPU`
   - 请记得先执行 `pip uninstall onnxruntime-gpu` and `pip uninstall onnxruntime-directml`。 接下来 `pip install onnxruntime-openvino --upgrade`。 
4. **简单的图形界面版本**：  
   - 体验简单易用的图形界面版本：  
      - [F5-TTS-ONNX GUI](https://github.com/patientx/F5-TTS-ONNX-gui)

5. **支持 NVIDIA TensorRT**：  
   - 针对 NVIDIA GPU 的 TensorRT 优化，请访问：  
      - [F5_TTS_Faster](https://github.com/WGS-note/F5_TTS_Faster)

6. **Download**
   - [Link](https://huggingface.co/H5N1AIDS/F5-TTS-ONNX/tree/main)

### 了解更多  
- 探索更多相关项目和资源：  
  [项目概览](https://github.com/DakeQQ?tab=repositories)

---  
