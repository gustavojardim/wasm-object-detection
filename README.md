# EN-US
# WASM Object Detection with YOLOv8 and wasi-nn

This project implements a WebAssembly (WASM) TCP server for real-time object detection using YOLOv8 and the `wasi-nn` API. The system is designed to support both CPU and GPU (CUDA) inference within isolated environments, facilitating portability and deployment via Kubernetes.

**Related Paper:** *[Insert Paper Title Here]* **Paper Abstract:** This work presents a high-performance object detection implementation using the Wasmtime runtime. The goal is to demonstrate the feasibility of using hardware acceleration (GPU) via the `wasi-nn` interface in WebAssembly modules, overcoming traditional performance limitations in sandboxed environments.

# README Structure

* **Basic Information:** Hardware and software requirements for replication.
* **Dependencies:** Specific versions of drivers and external libraries.
* **Installation:** Step-by-step guide to compile the module and export models.
* **Minimum Test:** A quick check of environment integrity.
* **Experiments:** Procedures to validate performance and functionality claims.
* **Troubleshooting:** Resolution matrix for CUDA linkage issues.

# Basic Information

### Execution Environment
* **Hardware (Used in tests):**
    * GPU: NVIDIA GeForce RTX 5060 Ti (Compute Capability 12.0)
    * CPU: [Specify your CPU, e.g., Intel i7 13th Gen]
    * RAM: Minimum 16GB recommended.
* **Software:**
    * Operating System: Linux (Tested on Ubuntu 22.04/24.04)
    * NVIDIA Driver: `590.48.01`
    * Wasmtime: `43.0.0`
    * Rust: `1.75+` with `wasm32-wasip2` target.

# Dependencies

* **Libtorch:** `2.10.0+cu130` (Essential for the inference backend).
* **Python:** `3.10+` (For export scripts and test clients).
* **Python Libraries:** `ultralytics`, `opencv-python`, `torch`.
* **Wasi-nn:** The Wasmtime runtime must be compiled/executed with neural network support enabled.

# Security Concerns

The WASM module requires privileged access to host resources through the `-S nn=y` flags (GPU/CPU access via libtorch) and `-S inherit-network=y` (opening TCP sockets). It is recommended to run the server in isolated networks or controlled containers, as the WebAssembly sandbox is partially relaxed to allow communication with the video driver.

# Installation

1.  **Model Export:**
    Convert the original YOLOv8 model to TorchScript:
    ```bash
    python3 scripts/export_yolo.py cpu
    python3 scripts/export_yolo.py cuda
    ```
    The files will be saved in the `models/` directory.

2.  **Rust Project Compilation:**
    Ensure the `wasm32-wasip2` target is installed and compile:
    ```bash
    cd inference
    cargo build --release --target wasm32-wasip2
    ```

# Minimum Test

To validate if the environment is correctly configured without performing heavy inference:

1.  Run the verification script:
    ```bash
    bash scripts/verify_setup.sh
    ```
2.  Start the server in CPU mode for communication testing:
    ```bash
    wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y \
    --dir ./models::/models \
    target/wasm32-wasip2/release/inference.wasm --device cpu
    ```
3.  In another terminal, run the test client:
    ```bash
    python3 test-scripts/test_tcp_client.py samples/image.png
    ```
    *Expected Outcome:* The server should return detection coordinates in the terminal.

# Experiments

## Claim #1: GPU Acceleration (CUDA) in WASM
Validates if the runtime can load CUDA kernels and perform inference on the RTX 5060 Ti GPU.

* **Configuration:** Use the `--device gpu` flag.
* **Command:**
    ```bash
    wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y --dir ./models::/models target/wasm32-wasip2/release/inference.wasm --device gpu
    ```
* **Expectation:** GPU usage should be visible via `nvidia-smi` or `nvtop`. Inference should be significantly faster than in CPU mode.

## Claim #2: Stability in Video Stream Processing
Validates the robustness of the TCP server under continuous frame flow.

* **Configuration:** Sending multiple frames through a video file.
* **Command:**
    ```bash
    python3 test_tcp_video.py samples/walking_people_hd.mp4 --save output_detection.mp4
    ```
* **Expected Time:** Processing should occur in real-time or faster, depending on the GPU load.
* **Result:** Generation of the `output_detection.mp4` file with rendered bounding boxes.

# PT-BR
# WASM Object Detection with YOLOv8 and wasi-nn

Este projeto implementa um servidor TCP em WebAssembly (WASM) para detecção de objetos em tempo real utilizando YOLOv8 e a API `wasi-nn`. O sistema foi desenvolvido para suportar inferência tanto em CPU quanto em GPU (CUDA) dentro de ambientes isolados, facilitando a portabilidade e o deploy via Kubernetes.

**Artigo Relacionado:** *[Insira aqui o Título do seu Artigo]*
**Resumo do Artigo:** Este trabalho apresenta uma implementação de detecção de objetos de alta performance utilizando o runtime Wasmtime. O objetivo é demonstrar a viabilidade do uso de aceleração de hardware (GPU) via interface `wasi-nn` em módulos WebAssembly, superando as limitações tradicionais de performance em ambientes sandboxed.

# Estrutura do readme.md

* **Informações Básicas:** Requisitos de hardware e software para replicação.
* **Dependências:** Versões específicas de drivers e bibliotecas externas.
* **Instalação:** Passo a passo para compilar o módulo e exportar modelos.
* **Teste Mínimo:** Verificação rápida da integridade do ambiente.
* **Experimentos:** Procedimentos para validar as reivindicações de performance e funcionalidade.
* **Troubleshooting:** Matriz de solução de problemas para linkage CUDA.

# Informações básicas

### Ambiente de Execução
* **Hardware (Utilizado nos testes):**
    * GPU: NVIDIA GeForce RTX 5060 Ti (Compute Capability 12.0)
    * CPU: [Especificar sua CPU, ex: Intel i7 13th Gen]
    * Memória RAM: Mínimo 16GB recomendado.
* **Software:**
    * Sistema Operacional: Linux (testado em Ubuntu 22.04/24.04)
    * NVIDIA Driver: `590.48.01`
    * Wasmtime: `43.0.0`
    * Rust: `1.75+` com target `wasm32-wasip2`

# Dependências

* **Libtorch:** `2.10.0+cu130` (Essencial para o backend de inferência).
* **Python:** `3.10+` (Para scripts de exportação e clientes de teste).
* **Bibliotecas Python:** `ultralytics`, `opencv-python`, `torch`.
* **Wasi-nn:** O runtime Wasmtime deve ser compilado/executado com suporte a redes neurais habilitado.

# Preocupações com segurança

O módulo WASM requer acesso privilegiado a recursos do host através das flags `-S nn=y` (acesso à GPU/CPU via libtorch) e `-S inherit-network=y` (abertura de sockets TCP). Recomenda-se executar o servidor em redes isoladas ou containers controlados, pois a sandbox do WebAssembly é parcialmente relaxada para permitir a comunicação com o driver de vídeo.

# Instalação

1.  **Exportação do Modelo:**
    Converta o modelo YOLOv8 original para TorchScript:
    ```bash
    python3 scripts/export_yolo.py cpu
    python3 scripts/export_yolo.py cuda
    ```
    Os arquivos serão salvos na pasta `models/`.

2.  **Compilação do Projeto Rust:**
    Certifique-se de que o target `wasm32-wasip2` está instalado e compile:
    ```bash
    cd inference
    cargo build --release --target wasm32-wasip2
    ```

# Teste mínimo

Para validar se o ambiente está configurado corretamente sem realizar inferências pesadas:

1.  Execute o script de verificação:
    ```bash
    bash scripts/verify_setup.sh
    ```
2.  Inicie o servidor em modo CPU para teste de comunicação:
    ```bash
    wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y \
    --dir ./models::/models \
    target/wasm32-wasip2/release/inference.wasm --device cpu
    ```
3.  Em outro terminal, execute o cliente de teste:
    ```bash
    python3 test-scripts/test_tcp_client.py samples/image.png
    ```
    *Expectativa:* O servidor deve retornar as coordenadas de detecção no terminal.

# Experimentos

## Reivindicação #1: Aceleração via GPU (CUDA) em WASM
Valida se o runtime consegue carregar os kernels CUDA e realizar a inferência na GPU RTX 5060 Ti.

* **Configuração:** Utilizar a flag `--device gpu`.
* **Comando:**
    ```bash
    wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y --dir ./models::/models target/wasm32-wasip2/release/inference.wasm --device gpu
    ```
* **Expectativa:** O uso de GPU deve ser visível via `nvidia-smi` ou `nvtop`. A inferência deve ser significativamente mais rápida que no modo CPU.

## Reivindicação #2: Estabilidade em Processamento de Vídeo
Valida a robustez do servidor TCP sob fluxo contínuo de frames.

* **Configuração:** Envio de múltiplos frames através de um arquivo de vídeo.
* **Comando:**
    ```bash
    python3 test_tcp_video.py samples/walking_people_hd.mp4 --save output_detection.mp4
    ```
* **Tempo Esperado:** O processamento deve ocorrer em tempo real ou superior, dependendo da carga da GPU.
* **Resultado:** Geração do arquivo `output_detection.mp4` com as bounding boxes renderizadas.

# LICENSE

MIT License
