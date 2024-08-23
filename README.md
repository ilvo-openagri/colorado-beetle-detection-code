# Coloradobeetles



## Getting started
To run the model on a NVIDIA GPU with CUDA, the pytorch packages need to be installed for the correct of the machine. To verify your CUDA version, please run

```Powershell
PS C:\Users\jbauwens\Documents\Code\openagri_ai> nvidia-smi.exe
Fri Aug 23 15:21:53 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 528.52       Driver Version: 528.52       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   48C    P0     9W /  30W |      0MiB /  2048MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

This output mentions that CUDA version 12.0 is used. To install the python dependencies with the correct CUDA version, change requirements.txt.

```
--extra-index-url https://download.pytorch.org/whl/cu121
```