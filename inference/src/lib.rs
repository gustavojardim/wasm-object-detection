use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, GenericImageView};
use serde::Serialize;

// Generate wasi-nn bindings from WIT  
wit_bindgen::generate!({
    path: "wit/deps/wasi-nn",
    world: "ml",
    default_bindings_module: "inference::bindings"
});

// The generated bindings create a `wasi` module
use self::wasi::nn::graph::{self, Graph, ExecutionTarget, GraphEncoding};
use self::wasi::nn::tensor::{Tensor, TensorType};

// YOLOv8n standard input size
const TARGET_SIZE: u32 = 640;
const TCP_ADDR: &str = "0.0.0.0:8080";

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

// Hardcoded COCO classes (Standard for YOLOv8)
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

fn read_exact_len_from_stream(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_json_result_to_stream(stream: &mut TcpStream, r: &serde_json::Value) -> Result<()> {
    let s = serde_json::to_vec(r)?;
    let len = (s.len() as u32).to_le_bytes();
    stream.write_all(&len)?;
    stream.write_all(&s)?;
    stream.flush()?;
    Ok(())
}

// Prepare image for YOLO (Resize 640x640 + Normalize 0-1)
fn img_to_nchw(img_bytes: &[u8]) -> Result<Vec<f32>> {
    let img = image::load_from_memory(img_bytes).context("decode image")?;
    let img = img.resize_exact(TARGET_SIZE, TARGET_SIZE, FilterType::Triangle);
    
    let (w, h) = img.dimensions();
    let mut nchw = vec![0f32; 3 * (w * h) as usize];

    for (x, y, pixel) in img.pixels() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        let idx = (y * w + x) as usize;
        let size = (w * h) as usize;
        
        nchw[idx] = r;
        nchw[idx + size] = g;
        nchw[idx + 2 * size] = b;
    }

    Ok(nchw)
}

fn load_graph(model_name: &str) -> Result<Graph> {
    let model_path = format!("/models/{}", model_name);
    let model_bytes = std::fs::read(&model_path)
        .with_context(|| format!("failed to read model at {}", model_path))?;
    
    // Try GPU first, fallback to CPU
    let graph = graph::load(
        &[model_bytes.clone()],
        GraphEncoding::Pytorch,
        ExecutionTarget::Gpu,
    ).or_else(|_| {
        eprintln!("[INFO] GPU load failed, falling back to CPU...");
        graph::load(
            &[model_bytes],
            GraphEncoding::Pytorch,
            ExecutionTarget::Cpu,
        )
    }).map_err(|e| anyhow!("Failed to load graph: {:?}", e))?;
    
    Ok(graph)
}

// Parse YOLO Output [1, 84, 8400] -> [cx, cy, w, h, class_probs...]
fn process_output(output: &[f32]) -> Vec<Det> {
    let num_elements = 8400; // Number of predictions
    
    let mut detections = Vec::new();

    for i in 0..num_elements {
        // Find the class with max score
        let mut max_score = 0.0;
        let mut best_class = 0;

        // Channels 4..84 are class probabilities
        for c in 0..80 {
            let idx = (4 + c) * num_elements + i; 
            let score = output[idx];
            if score > max_score {
                max_score = score;
                best_class = c;
            }
        }

        if max_score > 0.5 { // Confidence Threshold
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
    
    // Simple NMS (Non-Maximum Suppression)
    nms(detections, 0.45)
}

fn nms(mut dets: Vec<Det>, iou_thresh: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = Vec::new();

    while let Some(current) = dets.pop() {
        keep.push(current.clone());
        dets.retain(|other| box_iou(&current.bbox, &other.bbox) < iou_thresh);
    }
    keep
}

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

fn handle_client(mut stream: TcpStream, graph: &Graph) -> Result<()> {
    let peer = stream.peer_addr()?;
    eprintln!("[INFO] Client connected: {}", peer);

    let ctx = graph.init_execution_context()
        .map_err(|e| anyhow!("Failed to init execution context: {:?}", e))?;
    
    eprintln!("[DEBUG] Created execution context: {:?}", ctx);

    loop {
        // Read image data from client
        let img_bytes = match read_exact_len_from_stream(&mut stream) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("[INFO] Client {} disconnected: {}", peer, e);
                break;
            }
        };

        eprintln!("[DEBUG] Received {} bytes from {}", img_bytes.len(), peer);

        // Preprocess image
        let input = img_to_nchw(&img_bytes)?;

        // Prepare tensor data (convert f32 to bytes)
        let tensor_data: Vec<u8> = input
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Create tensor
        let tensor = Tensor::new(
            &vec![1, 3, TARGET_SIZE, TARGET_SIZE],
            TensorType::Fp32,
            &tensor_data,
        );

        // Run inference using wasi-nn - new API takes inputs and returns outputs
        let inputs = vec![("images".to_string(), tensor)];
        let outputs = ctx.compute(inputs)
            .map_err(|e| anyhow!("Failed to compute: {:?}", e))?;

        // Get first output tensor reference
        let output_tensor = &outputs.get(0)
            .ok_or_else(|| anyhow!("No output tensor"))?
            .1;
        
        // Convert bytes back to f32
        let output_floats: Vec<f32> = output_tensor
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Post-process detections
        let detections = process_output(&output_floats);
        eprintln!("[SUCCESS] Detected {} objects", detections.len());

        // Send result back to client
        let json = serde_json::to_value(&detections)?;
        write_json_result_to_stream(&mut stream, &json)?;
    }

    Ok(())
}

fn run_server() -> Result<()> {
    eprintln!("[INIT] Loading YOLOv8n model via wasi-nn...");
    let graph = load_graph("yolov8n_cuda.torchscript")?;
    eprintln!("[SUCCESS] Model loaded successfully");

    eprintln!("[INIT] Starting TCP server on {}...", TCP_ADDR);
    let listener = TcpListener::bind(TCP_ADDR)?;
    eprintln!("[SUCCESS] Server listening on {}", TCP_ADDR);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = handle_client(stream, &graph) {
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

#[no_mangle]
pub extern "C" fn _start() {
    if let Err(e) = run_server() {
        eprintln!("[FATAL] Server error: {:?}", e);
        std::process::exit(1);
    }
}
