// YOLOv8n Object Detection Server (TCP/UDP, WASI-NN)
// Optimized for Orin AGX (Zero-Copy, Throttled Cleanup)

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, GenericImageView};
use serde::Serialize;

#[macro_export]
macro_rules! info_log {
    ($($arg:tt)*) => {
        {
            let now = chrono::Local::now();
            eprintln!("[INFO {}] {}", now.format("%Y-%m-%d %H:%M:%S"), format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! debug_log {
    ($enabled:expr, $($arg:tt)*) => {
        if $enabled {
            let now = chrono::Local::now();
            eprintln!("[DEBUG {}] {}", now.format("%Y-%m-%d %H:%M:%S"), format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! error_log {
    ($($arg:tt)*) => {
        {
            let now = chrono::Local::now();
            eprintln!("[ERROR {}] {}", now.format("%Y-%m-%d %H:%M:%S"), format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! profile_log {
    ($enabled:expr, $obj:expr) => {
        if $enabled {
            let now = chrono::Local::now();
            if let Ok(line) = serde_json::to_string(&$obj) {
                println!("[PROFILE {}] {}", now.format("%Y-%m-%d %H:%M:%S"), line);
            }
        }
    };
}

// --- Constants ---
const TARGET_SIZE: u32 = 640; // Kept at 640 for Hybrid FP16/FP32 precision
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
    x1: f32, y1: f32, x2: f32, y2: f32,
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

/// Prepare image for YOLO.
/// OPTIMIZATION: Uses FilterType::Nearest for speed.
fn img_to_nchw(img_bytes: &[u8], nchw: &mut [f32]) -> Result<()> {
    let img = image::load_from_memory(img_bytes).context("decode image")?;
    let (w, h) = img.dimensions();
    
    // Resize only if necessary, use Nearest Neighbor for speed
    let img = if w == TARGET_SIZE && h == TARGET_SIZE {
        img
    } else {
        img.resize_exact(TARGET_SIZE, TARGET_SIZE, FilterType::Nearest)
    };
    
    let (w, h) = img.dimensions();
    let size = (w * h) as usize;
    debug_assert_eq!(nchw.len(), 3 * size);
    
    // Normalize to 0-1 and convert to Planar (RRR...GGG...BBB...)
    for (x, y, pixel) in img.pixels() {
        let idx = (y * w + x) as usize;
        nchw[idx] = pixel[0] as f32 / 255.0;
        nchw[idx + size] = pixel[1] as f32 / 255.0;
        nchw[idx + 2 * size] = pixel[2] as f32 / 255.0;
    }
    Ok(())
}

fn load_graph(device: &str, debug: bool) -> Result<Graph> {
    let (model_file, target) = match device.to_lowercase().as_str() {
        "cpu" => ("yolov8n_cpu.torchscript", ExecutionTarget::Cpu),
        "gpu" | "cuda" => ("yolov8n_cuda.torchscript", ExecutionTarget::Gpu),
        other => {
            error_log!("Unknown device '{}', defaulting to CPU.", other);
            ("yolov8n_cpu.torchscript", ExecutionTarget::Cpu)
        }
    };
    let model_path = format!("/models/{}", model_file);
    debug_log!(debug, "Selected device: {}, model: {}", device, model_file);
    let model_bytes = std::fs::read(&model_path)
        .with_context(|| format!("failed to read model at {}", model_path))?;
    debug_log!(debug, "Loading model: {} (target: {:?})", model_file, target);
    let graph = graph::load(&[model_bytes], GraphEncoding::Pytorch, target)
        .map_err(|e| anyhow!("Failed to load graph: {:?}", e))?;
    Ok(graph)
}

// --- YOLO Output Processing ---

fn process_output(output: &[f32]) -> Vec<Det> {
    let num_elements = 8400; 
    let mut detections = Vec::new();
    for i in 0..num_elements {
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

// --- Server Implementations ---

fn handle_client(mut stream: TcpStream, graph: &Graph, debug: bool) -> Result<()> {
    let peer = stream.peer_addr()?;
    info_log!("Client connected: {}", peer);
    let ctx = graph.init_execution_context()
        .map_err(|e| anyhow!("Failed to init execution context: {:?}", e))?;

    use std::time::Instant;
    let profile = debug;
    let size = (3 * TARGET_SIZE * TARGET_SIZE) as usize;
    let mut nchw = vec![0f32; size];

    #[derive(Default)]
    struct Metrics {
        tcp_read_ms: Vec<f64>,
        pre_ms: Vec<f64>,
        inf_ms: Vec<f64>,
        post_ms: Vec<f64>,
        tcp_write_ms: Vec<f64>,
    }
    let mut metrics = Metrics::default();

    loop {
        let t_tcp_read_start = Instant::now();
        let img_bytes = match read_exact_len_from_stream(&mut stream) {
            Ok(bytes) => bytes,
            Err(e) => {
                info_log!("Client {} disconnected: {}", peer, e);
                break;
            }
        };
        let t_tcp_read = t_tcp_read_start.elapsed();
        debug_log!(debug, "Received {} bytes from {} (TCP read: {:?})", img_bytes.len(), peer, t_tcp_read);

        let t_pre_start = Instant::now();
        img_to_nchw(&img_bytes, &mut nchw)?;
        let t_pre = t_pre_start.elapsed();

        let tensor_bytes: &[u8] = bytemuck::cast_slice(&nchw);
        let tensor = Tensor::new(&vec![1, 3, TARGET_SIZE, TARGET_SIZE], TensorType::Fp32, tensor_bytes);

        let t_inf_start = Instant::now();
        let inputs = vec![("images".to_string(), tensor)];
        let outputs = ctx.compute(inputs)
            .map_err(|e| anyhow!("Failed to compute: {:?}", e))?;
        let t_inf = t_inf_start.elapsed();

        let output_tensor = &outputs.get(0)
            .ok_or_else(|| anyhow!("No output tensor"))?
            .1;

        let t_post_start = Instant::now();
        let output_floats: Vec<f32> = output_tensor.data().chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
        let detections = process_output(&output_floats);
        let t_post = t_post_start.elapsed();

        debug_log!(debug, "Detected {} objects", detections.len());

        let t_tcp_write_start = Instant::now();
        let json = serde_json::to_value(&detections)?;
        write_json_result_to_stream(&mut stream, &json)?;
        let t_tcp_write = t_tcp_write_start.elapsed();

        metrics.tcp_read_ms.push((t_tcp_read.as_secs_f64() * 1000.0).round() / 1.0);
        metrics.pre_ms.push((t_pre.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
        metrics.inf_ms.push((t_inf.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
        metrics.post_ms.push((t_post.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
        metrics.tcp_write_ms.push((t_tcp_write.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
    }

    if !metrics.tcp_read_ms.is_empty() {
        let avg = |v: &Vec<f64>| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        let averages = serde_json::json!({
            "mode": "tcp",
            "avg_tcp_read_ms": avg(&metrics.tcp_read_ms),
            "avg_pre_ms": avg(&metrics.pre_ms),
            "avg_inf_ms": avg(&metrics.inf_ms),
            "avg_post_ms": avg(&metrics.post_ms),
            "avg_tcp_write_ms": avg(&metrics.tcp_write_ms),
            "requests": metrics.tcp_read_ms.len()
        });
        let _ = write_json_result_to_stream(&mut stream, &averages);
        profile_log!(profile, averages);
    }
    Ok(())
}

fn run_udp_server_with_addr(graph: &Graph, debug: bool, profile: bool, udp_addr: &str) -> Result<()> {
    let socket = UdpSocket::bind(udp_addr)?;
    info_log!("UDP server listening on {}", udp_addr);
    let mut buf = vec![0u8; 2048];
    let size = (3 * TARGET_SIZE * TARGET_SIZE) as usize;
    let mut nchw = vec![0f32; size];
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    let mut frames: HashMap<(u32, std::net::SocketAddr), (Instant, u16, Vec<Option<Vec<u8>>>)> = HashMap::new();
    let header_len = 8;
    let frame_timeout = Duration::from_secs(5);
    #[derive(Default)]
    struct Metrics {
        pre_ms: Vec<f64>,
        inf_ms: Vec<f64>,
        post_ms: Vec<f64>,
        output_kb: Vec<f64>,
        output_bytes: Vec<usize>,
        frames: usize,
    }
    let mut client_metrics: HashMap<std::net::SocketAddr, Metrics> = HashMap::new();
    let mut packet_count: u64 = 0;
    loop {
        let (len, src) = socket.recv_from(&mut buf)?;
        info_log!("Received UDP packet from {} ({} bytes)", src, len);
        // --- UDP HEALTH CHECK ---
        if len == 6 && &buf[..6] == b"HEALTH" {
            let _ = socket.send_to(b"OK", src);
            continue;
        }
        if len < header_len { continue; }
        let frame_id = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let chunk_idx = u16::from_be_bytes([buf[4], buf[5]]);
        let total_chunks = u16::from_be_bytes([buf[6], buf[7]]);
        let payload = &buf[header_len..len];
        if frame_id == 0 && chunk_idx == 0 && total_chunks == 0 {
            if let Some(metrics) = client_metrics.get(&src) {
                let stats = |v: &Vec<f64>| {
                    if v.is_empty() {
                        (0.0, 0.0, 0.0)
                    } else {
                        (v.iter().sum::<f64>() / v.len() as f64, *v.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(), *v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
                    }
                };
                let stats_usize = |v: &Vec<usize>| {
                    if v.is_empty() {
                        (0.0, 0.0, 0.0)
                    } else {
                        let v_f64: Vec<f64> = v.iter().map(|x| *x as f64).collect();
                        (v_f64.iter().sum::<f64>() / v_f64.len() as f64, *v_f64.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(), *v_f64.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
                    }
                };
                fn round_tuple(t: (f64, f64, f64)) -> (f64, f64, f64) {
                    (
                        (t.0 * 100.0).round() / 100.0,
                        (t.1 * 100.0).round() / 100.0,
                        (t.2 * 100.0).round() / 100.0,
                    )
                }
                let summary = serde_json::json!({
                    "mode": "udp",
                    "pre_ms": round_tuple(stats(&metrics.pre_ms)),
                    "inf_ms": round_tuple(stats(&metrics.inf_ms)),
                    "post_ms": round_tuple(stats(&metrics.post_ms)),
                    "output_kb": round_tuple(stats(&metrics.output_kb)),
                    "output_bytes": round_tuple(stats_usize(&metrics.output_bytes)),
                    "frames": metrics.frames
                });
                let _ = socket.send_to(&serde_json::to_vec(&summary).unwrap(), src);
                profile_log!(profile, summary);
                client_metrics.remove(&src);
                frames.retain(|&(_, addr), _| addr != src);
            }
            continue;
        }
        let key = (frame_id, src);
        let entry = frames.entry(key).or_insert_with(|| (Instant::now(), total_chunks, vec![None; total_chunks as usize]));
        entry.0 = Instant::now();
        if (chunk_idx as usize) < entry.2.len() {
            entry.2[chunk_idx as usize] = Some(payload.to_vec());
        }
        if entry.2.iter().all(|c| c.is_some()) {
            let mut img_bytes = Vec::with_capacity(entry.2.iter().map(|c| c.as_ref().unwrap().len()).sum());
            for chunk in &entry.2 {
                img_bytes.extend_from_slice(chunk.as_ref().unwrap());
            }
            frames.remove(&key);
            debug_log!(debug, "Reassembled frame {} ({} bytes) from {}", frame_id, img_bytes.len(), src);
            let t_pre_start = std::time::Instant::now();
            if let Err(e) = img_to_nchw(&img_bytes, &mut nchw) {
                error_log!("Decode: {}", e);
                continue;
            }
            let t_pre = t_pre_start.elapsed();
            let tensor_bytes: &[u8] = bytemuck::cast_slice(&nchw);
            let tensor = Tensor::new(&vec![1, 3, TARGET_SIZE, TARGET_SIZE], TensorType::Fp32, tensor_bytes);
            let t_inf_start = std::time::Instant::now();
            let ctx = match graph.init_execution_context() {
                Ok(c) => c,
                Err(e) => { error_log!("Context init: {:?}", e); continue; }
            };
            let inputs = vec![("images".to_string(), tensor)];
            let outputs = match ctx.compute(inputs) {
                Ok(o) => o,
                Err(e) => { error_log!("Compute: {:?}", e); continue; }
            };
            let t_inf = t_inf_start.elapsed();
            let t_post_start = std::time::Instant::now();
            let output_tensor = &outputs.get(0).ok_or_else(|| anyhow!("No output"))?.1;
            let output_floats: Vec<f32> = output_tensor.data().chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
            let detections = process_output(&output_floats);
            let t_post = t_post_start.elapsed();
            let json = match serde_json::to_vec(&detections) {
                Ok(j) => j,
                Err(e) => { error_log!("JSON: {}", e); continue; }
            };
            let _ = socket.send_to(&json, src);
            let json_size_kb = (json.len() as f64) / 1024.0;
            profile_log!(profile, serde_json::json!({
                "mode": "udp",
                "pre_ms": (t_pre.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "inf_ms": (t_inf.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "post_ms": (t_post.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
                "output_kb": (json_size_kb * 100.0).round() / 100.0,
                "output_bytes": json.len()
            }));
            let m = client_metrics.entry(src).or_default();
            m.pre_ms.push((t_pre.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
            m.inf_ms.push((t_inf.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
            m.post_ms.push((t_post.as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
            m.output_kb.push((json_size_kb * 100.0).round() / 100.0);
            m.output_bytes.push(json.len());
            m.frames += 1;
        }
        packet_count += 1;
        if packet_count % 100 == 0 {
            let now = Instant::now();
            frames.retain(|_, (ts, _, _)| now.duration_since(*ts) < frame_timeout);
        }
    }
    // unreachable: Ok(())
}

fn run_server() -> Result<()> {
    use std::env;
    let args: Vec<String> = env::args().collect();
    let mut device = "cpu"; // Default to CPU for safety
    let mut debug = false;
    let mut use_udp = false;
    let mut profile = false;
    let mut port: Option<u16> = None;
    for i in 0..args.len() {
        match args[i].as_str() {
            "--device" => if let Some(val) = args.get(i + 1) { device = val; },
            "--debug" => debug = true,
            "--udp" => use_udp = true,
            "--profile" => profile = true,
            "--port" => {
                if let Some(val) = args.get(i + 1) {
                    if let Ok(p) = val.parse::<u16>() {
                        port = Some(p);
                    }
                }
            },
            _ => {}
        }
    }
    let requested_device = device.to_lowercase();

    info_log!("Loading YOLOv8n (device: {})...", requested_device);
    let (graph, active_device) = match load_graph(&requested_device, debug) {
        Ok(g) => {
            info_log!("Model loaded for device: {}", requested_device);
            (g, requested_device.as_str())
        },
        Err(e) => {
            if requested_device != "cpu" {
                error_log!("[WARN] {} failed: {}. Fallback to CPU.", requested_device.to_uppercase(), e);
                let cpu_graph = load_graph("cpu", debug)?;
                info_log!("CPU model loaded as fallback.");
                (cpu_graph, "cpu")
            } else {
                return Err(anyhow!("Failed to load model: {:?}", e));
            }
        }
    };
    if active_device == "gpu" {
        info_log!("Warming up GPU...");
        if let Ok(ctx) = graph.init_execution_context() {
            let size = (3 * 640 * 640) as usize;
            let dummy_f32 = vec![0f32; size];
            let dummy_bytes: &[u8] = bytemuck::cast_slice(&dummy_f32);
            for _ in 0..3 {
                let tensor = Tensor::new(&vec![1, 3, 640, 640], TensorType::Fp32, dummy_bytes);
                let _ = ctx.compute(vec![("images".to_string(), tensor)]);
            }
            info_log!("GPU Warmup complete.");
        }
    } else if requested_device == "gpu" {
        info_log!("Skipping GPU warmup because active backend is CPU.");
    }
    if use_udp {
        let udp_port = port.unwrap_or(8081);
        let udp_addr = format!("0.0.0.0:{}", udp_port);
        run_udp_server_with_addr(&graph, debug, profile, &udp_addr)
    } else {
        let tcp_port = port.unwrap_or(8080);
        let tcp_addr = format!("0.0.0.0:{}", tcp_port);
        info_log!("Starting TCP server on {}...", tcp_addr);
        let listener = TcpListener::bind(tcp_addr)?;
        for stream in listener.incoming() {
            match stream {
                Ok(s) => if let Err(e) = handle_client(s, &graph, debug) { error_log!("Handler: {:?}", e); },
                Err(e) => error_log!("Connection: {:?}", e),
            }
        }
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn _start() {
    if let Err(e) = run_server() {
        error_log!("Server error: {:?}", e);
        std::process::exit(1);
    }
}