
//! YOLOv8n Object Detection Server (TCP/UDP, WASI-NN)
//!
//! - TCP: Receives images, returns detections as JSON.
//! - UDP: Receives images, returns detections as JSON.
//! - Model: YOLOv8n TorchScript (CPU/GPU)
//!
//! Args: --device [cpu|gpu] --debug --udp --profile

// --- Imports ---
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, GenericImageView};
use serde::Serialize;

// --- Logging Macros ---
#[macro_export]
macro_rules! debug_log {
    ($enabled:expr, $($arg:tt)*) => {
        if $enabled { eprintln!("[DEBUG] {}", format!($($arg)*)); }
    };
}

#[macro_export]
macro_rules! profile_log {
    ($enabled:expr, $($arg:tt)*) => {
        if $enabled { eprintln!("[PROFILE] {}", format!($($arg)*)); }
    };
}

// --- Constants ---
const TARGET_SIZE: u32 = 640; // YOLOv8n standard input size
const TCP_ADDR: &str = "0.0.0.0:8080";
const UDP_ADDR: &str = "0.0.0.0:8081";

// --- Model Classes (COCO) ---
const CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

// --- Data Structures ---
#[derive(Serialize, Debug, Clone)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Serialize, Debug, Clone)]
struct Det {
    class_id: usize,
    class: String,
    confidence: f32,
    bbox: BBox,
}

// --- WASI-NN Bindings ---
wit_bindgen::generate!({
    path: "wit/deps/wasi-nn",
    world: "ml",
    default_bindings_module: "inference::bindings"
});
use self::wasi::nn::graph::{self, Graph, ExecutionTarget, GraphEncoding};
use self::wasi::nn::tensor::{Tensor, TensorType};

// --- Utility Functions ---

/// Read a length-prefixed buffer from a TCP stream
fn read_exact_len_from_stream(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

/// Write a JSON value to a TCP stream with length prefix
fn write_json_result_to_stream(stream: &mut TcpStream, r: &serde_json::Value) -> Result<()> {
    let s = serde_json::to_vec(r)?;
    let len = (s.len() as u32).to_le_bytes();
    stream.write_all(&len)?;
    stream.write_all(&s)?;
    stream.flush()?;
    Ok(())
}

/// Prepare image for YOLO (resize 640x640 if needed, normalize 0-1, NCHW) with buffer reuse
fn img_to_nchw(img_bytes: &[u8], nchw: &mut [f32]) -> Result<()> {
    let img = image::load_from_memory(img_bytes).context("decode image")?;
    let (w, h) = img.dimensions();
    let img = if w == TARGET_SIZE && h == TARGET_SIZE {
        img
    } else {
        img.resize_exact(TARGET_SIZE, TARGET_SIZE, FilterType::Triangle)
    };
    let (w, h) = img.dimensions();
    let size = (w * h) as usize;
    debug_assert_eq!(nchw.len(), 3 * size);
    for (x, y, pixel) in img.pixels() {
        let idx = (y * w + x) as usize;
        nchw[idx] = pixel[0] as f32 / 255.0;
        nchw[idx + size] = pixel[1] as f32 / 255.0;
        nchw[idx + 2 * size] = pixel[2] as f32 / 255.0;
    }
    Ok(())
}

/// Load YOLOv8n model graph for the given device
fn load_graph(device: &str, debug: bool) -> Result<Graph> {
    let (model_file, target) = match device {
        "cpu" => ("yolov8n_cpu.torchscript", ExecutionTarget::Cpu),
        _ => ("yolov8n_cuda.torchscript", ExecutionTarget::Gpu),
    };
    let model_path = format!("/models/{}", model_file);
    let model_bytes = std::fs::read(&model_path)
        .with_context(|| format!("failed to read model at {}", model_path))?;
    if debug {
        eprintln!("[DEBUG] Loading model: {} (target: {:?})", model_file, target);
    }
    let graph = graph::load(
        &[model_bytes],
        GraphEncoding::Pytorch,
        target,
    ).map_err(|e| anyhow!("Failed to load graph: {:?}", e))?;
    Ok(graph)
}

// --- YOLO Output Processing ---

/// Parse YOLO output tensor into detection structs, apply NMS
fn process_output(output: &[f32]) -> Vec<Det> {
    let num_elements = 8400; // Number of predictions
    let mut detections = Vec::new();
    for i in 0..num_elements {
        // Find the class with max score
        let (mut max_score, mut best_class) = (0.0, 0);
        for c in 0..80 {
            let idx = (4 + c) * num_elements + i;
            let score = output[idx];
            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }
        if max_score > 0.5 {
            let cx = output[0 * num_elements + i];
            let cy = output[1 * num_elements + i];
            let w  = output[2 * num_elements + i];
            let h  = output[3 * num_elements + i];
            // Convert center-wh to corner coordinates normalized (0.0-1.0)
            let x1 = (cx - w / 2.0) / TARGET_SIZE as f32;
            let y1 = (cy - h / 2.0) / TARGET_SIZE as f32;
            let x2 = (cx + w / 2.0) / TARGET_SIZE as f32;
            let y2 = (cy + h / 2.0) / TARGET_SIZE as f32;
            detections.push(Det {
                class_id: best_class,
                class: CLASSES.get(best_class).unwrap_or(&"?").to_string(),
                confidence: max_score,
                bbox: BBox { x1, y1, x2, y2 },
            });
        }
    }
    nms(detections, 0.45)
}

/// Non-Maximum Suppression (NMS)
fn nms(mut dets: Vec<Det>, iou_thresh: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = Vec::new();
    while let Some(current) = dets.pop() {
        keep.push(current.clone());
        dets.retain(|other| box_iou(&current.bbox, &other.bbox) < iou_thresh);
    }
    keep
}

/// Intersection-over-Union (IoU) for two bounding boxes
fn box_iou(b1: &BBox, b2: &BBox) -> f32 {
    let xi1 = b1.x1.max(b2.x1);
    let yi1 = b1.y1.max(b2.y1);
    let xi2 = b1.x2.min(b2.x2);
    let yi2 = b1.y2.min(b2.y2);
    let inter_w = (xi2 - xi1).max(0.0);
    let inter_h = (yi2 - yi1).max(0.0);
    let inter_area = inter_w * inter_h;
    let b1_area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    let b2_area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
    let union_area = b1_area + b2_area - inter_area;
    if union_area == 0.0 { 0.0 } else { inter_area / union_area }
}

// --- Server Implementations ---

/// Handle a single TCP client connection
fn handle_client(mut stream: TcpStream, graph: &Graph, debug: bool) -> Result<()> {
    let peer = stream.peer_addr()?;
    eprintln!("[INFO] Client connected: {}", peer);
    let ctx = graph.init_execution_context()
        .map_err(|e| anyhow!("Failed to init execution context: {:?}", e))?;
    debug_log!(debug, "Created execution context: {:?}", ctx);
    use std::time::Instant;
    let profile = debug; // For now, profile logs follow debug, can be separated if needed
    // Pre-allocate buffers
    let size = (3 * TARGET_SIZE * TARGET_SIZE) as usize;
    let mut nchw = vec![0f32; size];
    let mut tensor_data = vec![0u8; size * 4];
    loop {
        // --- TCP Read ---
        let t_tcp_read_start = Instant::now();
        let img_bytes = match read_exact_len_from_stream(&mut stream) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("[INFO] Client {} disconnected: {}", peer, e);
                break;
            }
        };
        let t_tcp_read = t_tcp_read_start.elapsed();
        debug_log!(debug, "Received {} bytes from {} (TCP read: {:?})", img_bytes.len(), peer, t_tcp_read);
        // --- Preprocessing ---
        let t_pre_start = Instant::now();
        img_to_nchw(&img_bytes, &mut nchw)?;
        let t_pre = t_pre_start.elapsed();
        // Prepare tensor data (convert f32 to bytes) using pre-allocated buffer
        for (i, f) in nchw.iter().enumerate() {
            tensor_data[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
        }
        // Create tensor
        let tensor = Tensor::new(&vec![1, 3, TARGET_SIZE, TARGET_SIZE], TensorType::Fp32, &tensor_data);
        // --- Inference ---
        let t_inf_start = Instant::now();
        let inputs = vec![("images".to_string(), tensor)];
        let outputs = ctx.compute(inputs)
            .map_err(|e| anyhow!("Failed to compute: {:?}", e))?;
        let t_inf = t_inf_start.elapsed();
        // Get first output tensor reference
        let output_tensor = &outputs.get(0)
            .ok_or_else(|| anyhow!("No output tensor"))?
            .1;
        // --- Postprocessing ---
        let t_post_start = Instant::now();
        // Convert bytes back to f32
        let output_floats: Vec<f32> = output_tensor.data().chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
        let detections = process_output(&output_floats);
        let t_post = t_post_start.elapsed();
        debug_log!(debug, "Detected {} objects", detections.len());
        // --- TCP Write ---
        let t_tcp_write_start = Instant::now();
        let json = serde_json::to_value(&detections)?;
        write_json_result_to_stream(&mut stream, &json)?;
        let t_tcp_write = t_tcp_write_start.elapsed();
        profile_log!(profile, "TCP read: {:?}, Preprocessing: {:?}, Inference: {:?}, Postprocessing: {:?}, TCP write: {:?}",
            t_tcp_read, t_pre, t_inf, t_post, t_tcp_write);
    }
    Ok(())
}

/// UDP server: receive image, run inference, send JSON result
fn run_udp_server(graph: &Graph, debug: bool, profile: bool) -> Result<()> {
    let socket = UdpSocket::bind(UDP_ADDR)?;
    eprintln!("[SUCCESS] UDP server listening on {}", UDP_ADDR);
    let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
    let size = (3 * TARGET_SIZE * TARGET_SIZE) as usize;
    let mut nchw = vec![0f32; size];
    let mut tensor_data = vec![0u8; size * 4];
    loop {
        let (len, src) = socket.recv_from(&mut buf)?;
        debug_log!(debug, "Received {} bytes from {}", len, src);
        let img_bytes = &buf[..len];
        // Preprocessing
        let t_pre_start = std::time::Instant::now();
        if let Err(e) = img_to_nchw(img_bytes, &mut nchw) {
            eprintln!("[ERROR] Image decode error: {}", e);
            continue;
        }
        let t_pre = t_pre_start.elapsed();
        // Prepare tensor data using pre-allocated buffer
        for (i, f) in nchw.iter().enumerate() {
            tensor_data[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
        }
        let tensor = Tensor::new(&vec![1, 3, TARGET_SIZE, TARGET_SIZE], TensorType::Fp32, &tensor_data);
        // Inference
        let t_inf_start = std::time::Instant::now();
        let ctx = match graph.init_execution_context() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[ERROR] Failed to init execution context: {:?}", e);
                continue;
            }
        };
        let inputs = vec![("images".to_string(), tensor)];
        let outputs = match ctx.compute(inputs) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[ERROR] Failed to compute: {:?}", e);
                continue;
            }
        };
        let t_inf = t_inf_start.elapsed();
        // Postprocessing
        let t_post_start = std::time::Instant::now();
        let output_tensor = &outputs.get(0).ok_or_else(|| anyhow!("No output tensor"))?.1;
        let output_floats: Vec<f32> = output_tensor.data().chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
        let detections = process_output(&output_floats);
        let t_post = t_post_start.elapsed();
        debug_log!(debug, "Detected {} objects", detections.len());
        // Send result back as JSON
        let json = match serde_json::to_vec(&detections) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[ERROR] JSON encode error: {}", e);
                continue;
            }
        };
        let _ = socket.send_to(&json, src);
        profile_log!(profile, "UDP Preprocessing: {:?}, Inference: {:?}, Postprocessing: {:?}, UDP write: {} bytes", t_pre, t_inf, t_post, json.len());
    }
}

/// Main server entrypoint: parses args, loads model, starts TCP/UDP server
fn run_server() -> Result<()> {
    use std::env;
    let args: Vec<String> = env::args().collect();
    let mut device = "gpu";
    let mut debug = false;
    let mut use_udp = false;
    let mut profile = false;
    for i in 0..args.len() {
        match args[i].as_str() {
            "--device" => {
                if let Some(val) = args.get(i + 1) {
                    if val == "cpu" || val == "gpu" {
                        device = val;
                    }
                }
            },
            "--debug" => debug = true,
            "--udp" => use_udp = true,
            "--profile" => profile = true,
            _ => {}
        }
    }
    eprintln!("[INIT] Loading YOLOv8n model via wasi-nn (device: {})...", device);
    let graph = match load_graph(device, debug) {
        Ok(g) => {
            eprintln!("[SUCCESS] Model loaded for device: {}", device);
            g
        },
        Err(e) => {
            if device == "gpu" {
                eprintln!("[WARN] Failed to load GPU model: {}. Falling back to CPU model...", e);
                let g = load_graph("cpu", debug)
                    .map_err(|e| anyhow!("Failed to load fallback CPU model: {:?}", e))?;
                eprintln!("[SUCCESS] Model loaded for device: cpu");
                g
            } else {
                return Err(anyhow!("Failed to load model: {:?}", e));
            }
        }
    };
    if use_udp {
        eprintln!("[INIT] Starting UDP server on {}...", UDP_ADDR);
        run_udp_server(&graph, debug, profile)
    } else {
        eprintln!("[INIT] Starting TCP server on {}...", TCP_ADDR);
        let listener = TcpListener::bind(TCP_ADDR)?;
        eprintln!("[SUCCESS] Server listening on {}", TCP_ADDR);
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    if let Err(e) = handle_client(stream, &graph, debug) {
                        eprintln!("[ERROR] Client handler error: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("[ERROR] Connection failed: {:?}", e);
                }
            }
        }
        Ok(())
    }
}

// --- Entrypoint ---
#[no_mangle]
pub extern "C" fn _start() {
    if let Err(e) = run_server() {
        eprintln!("[FATAL] Server error: {:?}", e);
        std::process::exit(1);
    }
}
