use std::io::{self, Read, Write};
use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, GenericImageView};
use serde::Serialize;
use wasi_nn::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding, TensorType};

// YOLOv8n standard input size
const TARGET_SIZE: u32 = 640;

#[derive(Serialize, Debug, Clone)]
struct Det {
    class_id: usize,
    class_name: String,
    score: f32,
    bbox: [f32; 4], // [x, y, w, h] normalized
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

fn read_exact_len() -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    io::stdin().read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    io::stdin().read_exact(&mut buf)?;
    Ok(buf)
}

fn write_json_result(r: &serde_json::Value) -> Result<()> {
    let s = serde_json::to_vec(r)?;
    let len = (s.len() as u32).to_le_bytes();
    let mut out = io::stdout();
    out.write_all(&len)?;
    out.write_all(&s)?;
    out.flush()?;
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
    GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::GPU)
        .build_from_bytes(&[&model_bytes])
        .or_else(|_| {
            eprintln!("[INFO] GPU load failed, falling back to CPU...");
            GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
                .build_from_bytes(&[&model_bytes])
        })
        .map_err(|e| anyhow!("Failed to load graph: {:?}", e))
}

// Parse YOLO Output [1, 84, 8400] -> [cx, cy, w, h, class_probs...]
fn process_output(output: &[f32]) -> Vec<Det> {
    let num_elements = 8400; // Number of predictions
    
    // The output is usually flattened, representing shape [1, 84, 8400]
    // We iterate "columns" (predictions)
    
    let mut detections = Vec::new();

    for i in 0..num_elements {
        // Find the class with max score
        let mut max_score = 0.0;
        let mut best_class = 0;

        // Channels 4..84 are class probabilities
        for c in 0..80 {
            // Index logic: channel * width + column
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

            // Convert center-wh to top-left-wh normalized (0.0-1.0)
            let x = (cx - w / 2.0) / TARGET_SIZE as f32;
            let y = (cy - h / 2.0) / TARGET_SIZE as f32;
            let w = w / TARGET_SIZE as f32;
            let h = h / TARGET_SIZE as f32;

            detections.push(Det {
                class_id: best_class,
                class_name: CLASSES.get(best_class).unwrap_or(&"?").to_string(),
                score: max_score,
                bbox: [x, y, w, h],
            });
        }
    }
    
    // Simple NMS (Non-Maximum Suppression)
    nms(detections, 0.45)
}

fn nms(mut dets: Vec<Det>, iou_thresh: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep = Vec::new();

    while let Some(current) = dets.pop() {
        keep.push(current.clone());
        dets.retain(|other| iou(&current.bbox, &other.bbox) < iou_thresh);
    }
    keep
}

fn iou(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let (x1, y1, w1, h1) = (b1[0], b1[1], b1[2], b1[3]);
    let (x2, y2, w2, h2) = (b2[0], b2[1], b2[2], b2[3]);

    let xi1 = x1.max(x2);
    let yi1 = y1.max(y2);
    let xi2 = (x1 + w1).min(x2 + w2);
    let yi2 = (y1 + h1).min(y2 + h2);

    let inter_w = (xi2 - xi1).max(0.0);
    let inter_h = (yi2 - yi1).max(0.0);
    let inter_area = inter_w * inter_h;

    let b1_area = w1 * h1;
    let b2_area = w2 * h2;
    let union_area = b1_area + b2_area - inter_area;

    if union_area == 0.0 { 0.0 } else { inter_area / union_area }
}

fn run_loop() -> Result<()> {
    eprintln!("[DEBUG] Loading YOLOv8n...");
    let graph = load_graph("yolov8n.torchscript")?; 
    let mut ctx = graph.init_execution_context()?;

    eprintln!("[SUCCESS] Ready for inference");

    loop {
        let img_bytes = read_exact_len()?;
        let input = img_to_nchw(&img_bytes)?;

        let dims = [1, 3, TARGET_SIZE as usize, TARGET_SIZE as usize];
        ctx.set_input(0, TensorType::F32, &dims, &input)?;
        ctx.compute()?;

        // Output buffer size: 1 * 84 * 8400 * 4 bytes ≈ 2.8 MB
        let mut output_buffer = vec![0f32; 1 * 84 * 8400]; 
        let bytes_written = ctx.get_output(0, &mut output_buffer)?;
        
        // Truncate to actual data length
        let elements = bytes_written / std::mem::size_of::<f32>();
        output_buffer.truncate(elements);

        let detections = process_output(&output_buffer);
        let json = serde_json::to_value(&detections)?;
        write_json_result(&json)?;
    }
}

#[no_mangle]
pub extern "C" fn _start() {
    if let Err(e) = run_loop() {
        eprintln!("[ERROR] {:?}", e);
    }
}