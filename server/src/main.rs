use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use futures::{SinkExt, StreamExt};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use wasmtime::*;
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi_nn::witx::WasiNnCtx;

// Combined context that holds both WASI and WASI-NN
struct HostContext {
    wasi: wasmtime_wasi::WasiCtx,
    wasi_nn: WasiNnCtx,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Path to WASM module
    let wasm_path = "../inference/target/wasm32-wasip1/release/wasm_inference.wasm";
    
    // Create Wasmtime engine
    let engine = Engine::default();
    
    // Load the WASM module
    let module = Module::from_file(&engine, wasm_path)?;

    // Create WASI context with stdio and directory access
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .preopened_dir(
            wasmtime_wasi::sync::Dir::open_ambient_dir("../models", wasmtime_wasi::sync::ambient_authority()),
            "/models",
            wasmtime_wasi::sync::DirPerms::all(),
            wasmtime_wasi::sync::FilePerms::all(),
        )?
        .build();

    // Create WASI-NN context
    let wasi_nn = WasiNnCtx::new(vec![], wasmtime_wasi_nn::backend::BackendRegistry::default());

    // Create host context combining both
    let host_ctx = HostContext { wasi, wasi_nn };
    
    // Create store with host context
    let mut store = Store::new(&engine, host_ctx);
    
    // Create linker and add WASI + WASI-NN
    let mut linker = Linker::new(&engine);
    wasmtime_wasi::add_to_linker(&mut linker, |ctx: &mut HostContext| &mut ctx.wasi)?;
    wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |ctx: &mut HostContext| &mut ctx.wasi_nn)?;

    // Instantiate the module
    let instance = linker.instantiate(&mut store, &module)?;
    
    // Get the _start function
    let start_func = instance.get_typed_func::<(), ()>(&mut store, "_start")?;

    // Wrap store and instance for shared access
    let wasm_state = Arc::new(Mutex::new((store, instance, start_func)));

    // Spawn background task to call _start
    let wasm_clone = Arc::clone(&wasm_state);
    tokio::task::spawn_blocking(move || {
        let mut state = wasm_clone.blocking_lock();
        let (ref mut store, _, ref start) = *state;
        if let Err(e) = start.call(store, ()) {
            eprintln!("WASM _start error: {:?}", e);
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
