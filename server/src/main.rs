use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use futures::{SinkExt, StreamExt};
use anyhow::Result;
use std::{fs::File, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use wasmtime::*;
use wasi_common::{WasiCtx, sync::Dir};
use wasmtime_wasi_nn::witx::WasiNnCtx;
use std::io::{Read, Write};
use wasi_common::pipe::{ReadPipe, WritePipe};

struct StoreState {
    wasi: WasiCtx,
    wasi_nn_witx: WasiNnCtx,
}

// FIX: Use the specific os_pipe types, not std::fs::File
struct WasmBridge {
    input_writer: os_pipe::PipeWriter, 
    output_reader: os_pipe::PipeReader, 
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("[DEBUG] Server starting...");
    let server_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    
    // Path to WASM module
    let wasm_path = server_dir.join("inference.wasm");
    println!("[DEBUG] WASM module path: {}", wasm_path.display());
    
    // Create Wasmtime engine
    let engine = Engine::default();
    println!("[DEBUG] Wasmtime engine created");
    
    // Load the WASM module
    let module = Module::from_file(&engine, wasm_path)?;
    println!("[DEBUG] WASM module loaded successfully");

    let models_dir = Dir::from_std_file(File::open(server_dir.join("../models"))?);

    // 1. Create Pipes
    // Host writes to input_writer -> Guest reads from guest_stdin_reader
    let (guest_stdin_reader, host_stdin_writer) = os_pipe::pipe()?;
    // Guest writes to guest_stdout_writer -> Host reads from output_reader
    let (output_reader, guest_stdout_writer) = os_pipe::pipe()?;

    let stdin = ReadPipe::new(guest_stdin_reader);
    let stdout = WritePipe::new(guest_stdout_writer);

    let wasi = wasi_common::sync::WasiCtxBuilder::new()
        .inherit_stderr()
        .stdin(Box::new(stdin))
        .stdout(Box::new(stdout))
        .preopened_dir(models_dir, "/models")?
        .build();

    let graphs: Vec<(String, String)> = vec![];
    let (backends, registry) = wasmtime_wasi_nn::preload(&graphs)?;
    let wasi_nn = WasiNnCtx::new(backends, registry);

    
    let mut store = Store::new(
        &engine,
        StoreState {
            wasi,
            wasi_nn_witx: wasi_nn,
        },
    );

    let mut linker: Linker<StoreState> = Linker::new(&engine);
    wasi_common::sync::add_to_linker(&mut linker, |s| &mut s.wasi)?;
    wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |s| &mut s.wasi_nn_witx)?;

    
    let instance = linker.instantiate(&mut store, &module)?;
    let start_func = instance.get_typed_func::<(), ()>(&mut store, "_start")?;

    // 3. Spawn WASM Guest
    std::thread::spawn(move || {
        println!("[GUEST] Starting WASM loop...");
        if let Err(e) = start_func.call(&mut store, ()) {
            eprintln!("[GUEST] Error: {:?}", e);
        }
    });

    // 4. Create Bridge
    let bridge = Arc::new(Mutex::new(WasmBridge {
        input_writer: host_stdin_writer,
        output_reader: output_reader,
    }));

    // 5. Start WebSocket Server
    let listener = TcpListener::bind("127.0.0.1:9001").await?;
    println!("WebSocket server listening on ws://127.0.0.1:9001");

    loop {
        let (stream, _) = listener.accept().await?;
        let bridge_clone = bridge.clone();
        
        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, bridge_clone).await {
                eprintln!("Client error: {:?}", e);
            }
        });
    }
}

async fn handle_client(
    stream: tokio::net::TcpStream,
    bridge: Arc<Mutex<WasmBridge>>,
) -> Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    println!("Client connected");

    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(m) if m.is_binary() => {
                let img_bytes = m.into_data();
                println!("[HOST] Received {} bytes from client", img_bytes.len());

                let response_json = {
                    let bridge = bridge.clone();
                    let img_bytes = img_bytes.clone();
                    
                    tokio::task::spawn_blocking(move || -> Result<String> {
                        let mut guard = bridge.blocking_lock();
                        
                        // 1. Write Length
                        let len_bytes = (img_bytes.len() as u32).to_le_bytes();
                        guard.input_writer.write_all(&len_bytes)?;
                        
                        // 2. Write Data
                        guard.input_writer.write_all(&img_bytes)?;
                        guard.input_writer.flush()?;
                        
                        // 3. Read Response Length
                        let mut resp_len_buf = [0u8; 4];
                        guard.output_reader.read_exact(&mut resp_len_buf)?;
                        let resp_len = u32::from_le_bytes(resp_len_buf) as usize;

                        // 4. Read Response JSON
                        let mut resp_buf = vec![0u8; resp_len];
                        guard.output_reader.read_exact(&mut resp_buf)?;
                        
                        let s = String::from_utf8(resp_buf)?;
                        Ok(s)
                    }).await??
                };

                println!("[HOST] Received result from WASM: {}", response_json);
                ws_sender.send(tokio_tungstenite::tungstenite::Message::Text(response_json)).await?;
            }
            Ok(_) => {}
            Err(_e) => break, // FIX: Added underscore
        }
    }
    Ok(())
}