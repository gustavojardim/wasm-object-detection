use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use futures::{SinkExt, StreamExt};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use wasmtime::*;
use wasi_common::WasiCtx;
use wasmtime_wasi_nn::witx::WasiNnCtx;

// Combined context that holds both WASI and WASI-NN
struct StoreState {
    wasi: WasiCtx,
    wasi_nn_witx: WasiNnCtx,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("[DEBUG] Server starting...");
    
    // Path to WASM module
    let wasm_path = "../inference/target/wasm32-wasip1/release/wasm_inference.wasm";
    println!("[DEBUG] WASM module path: {}", wasm_path);
    
    // Create Wasmtime engine
    let engine = Engine::default();
    println!("[DEBUG] Wasmtime engine created");
    
    // Load the WASM module
    let module = Module::from_file(&engine, wasm_path)?;
    println!("[DEBUG] WASM module loaded successfully");

    // Create a dir for models access
    let models_dir = wasi_common::sync::Dir::open_ambient_dir(
        "../models", 
        wasi_common::sync::ambient_authority()
    )?;
    println!("[DEBUG] Models directory opened: ../models");

    // Create WASI context with stdio and directory access
    let wasi = wasi_common::sync::WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_env()?
        .preopened_dir(models_dir, "/models")?
        .inherit_args()?
        .build();
    println!("[DEBUG] WASI context created with /models mapped");

    // Create WASI-NN context
    let graphs: Vec<(String, String)> = vec![];
    let (backends, registry) = wasmtime_wasi_nn::preload(&graphs)?;
    let wasi_nn = WasiNnCtx::new(backends, registry);
    println!("[DEBUG] WASI-NN context created");

    // Create store with state
    let mut store = Store::new(
        &engine,
        StoreState {
            wasi,
            wasi_nn_witx: wasi_nn,
        },
    );
    println!("[DEBUG] Store created with combined state");
    
    // Create linker and add WASI + WASI-NN
    let mut linker: Linker<StoreState> = Linker::new(&engine);
    wasi_common::sync::add_to_linker(&mut linker, |state: &mut StoreState| &mut state.wasi)?;
    wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |state: &mut StoreState| &mut state.wasi_nn_witx)?;
    println!("[DEBUG] Linker created with WASI and WASI-NN");

    // Instantiate the module
    let instance = linker.instantiate(&mut store, &module)?;
    println!("[DEBUG] WASM module instantiated");
    
    // Get the _start function
    let start_func = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
    println!("[DEBUG] _start function retrieved");

    // Wrap store and instance for shared access
    let wasm_state = Arc::new(Mutex::new((store, instance, start_func)));

    // Spawn background task to call _start
    println!("[DEBUG] Spawning background task to execute _start...");
    let wasm_clone = Arc::clone(&wasm_state);
    tokio::task::spawn_blocking(move || {
        println!("[DEBUG] Background task started, calling _start...");
        let mut state = wasm_clone.blocking_lock();
        let (ref mut store, _, ref start) = *state;
        if let Err(e) = start.call(store, ()) {
            eprintln!("[ERROR] WASM _start error: {:#?}", e);
        }
    });

    // Give the WASM module a moment to initialize
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Start WebSocket server
    let listener = TcpListener::bind("127.0.0.1:9001").await?;
    println!("WebSocket server listening on ws://127.0.0.1:9001");

    loop {
        let (stream, _) = listener.accept().await?;

        tokio::spawn(async move {
            if let Err(e) = handle_client(stream).await {
                eprintln!("client error: {:?}", e);
            }
        });
    }
}

async fn handle_client(
    stream: tokio::net::TcpStream,
) -> Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    println!("client connected");

    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(m) if m.is_binary() => {
                let _img_bytes = m.into_data();
                
                // For now, return a placeholder response
                // The WASM module is running in background and reads from stdin
                let response = serde_json::json!([
                    {
                        "class": "car",
                        "score": 0.95,
                        "bbox": [0.1, 0.2, 0.4, 0.3]
                    }
                ]);
                
                let json_str = serde_json::to_string(&response)?;
                ws_sender.send(tokio_tungstenite::tungstenite::Message::Text(json_str)).await?;
            }
            Ok(m) if m.is_text() => {
                eprintln!("text messages not yet implemented");
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("websocket error: {:?}", e);
                break;
            }
        }
    }

    println!("client disconnected");
    Ok(())
}
