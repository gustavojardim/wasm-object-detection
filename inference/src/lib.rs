use std::{
    fs,
    io::{self, Read, Write},
};
use anyhow::{anyhow, Context, Result};
use image::{self, imageops::FilterType};
use serde::Serialize;
use wasi_nn::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, TensorType};

const TARGET_H: u32 = 480;
const TARGET_W: u32 = 640;

#[derive(Serialize)]
struct Det {
    class: String,
    score: f32,
    bbox: [f32; 4],
}

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

fn img_to_nchw(img_bytes: &[u8]) -> Result<Vec<f32>> {
    let img = image::load_from_memory(img_bytes).context("decode image")?;
    let img = img
        .resize_exact(TARGET_W, TARGET_H, FilterType::Lanczos3)
        .to_rgb8();

    let (w, h) = (TARGET_W as usize, TARGET_H as usize);
    let mut nchw = vec![0f32; 3 * w * h];

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                nchw[c * w * h + y * w + x] = v;
            }
        }
    }

    Ok(nchw)
}

fn load_graph(model_name: &str) -> Result<Graph> {
    let model_path = format!("/models/{}", model_name);
    let model_bytes = fs::read(&model_path)
        .with_context(|| format!("failed to read model at {}", model_path))?;

    eprintln!(
        "[wasi-nn] loading graph: path={}, bytes={}",
        model_path,
        model_bytes.len()
    );

    let try_gpu = GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::GPU)
        .build_from_bytes(&[&model_bytes]);

    match try_gpu {
        Ok(g) => {
            eprintln!("[wasi-nn] graph loaded on GPU");
            Ok(g)
        }
        Err(e) => {
            eprintln!(
                "[wasi-nn] GPU load failed: {:?} (falling back to CPU)",
                e
            );
            GraphBuilder::new(GraphEncoding::Pytorch, ExecutionTarget::CPU)
                .build_from_bytes(&[&model_bytes])
                .map_err(|err| anyhow!("failed to load PyTorch graph: {:?}", err))
        }
    }
}

fn run_inference(ctx: &mut GraphExecutionContext<'_>, input: &[f32]) -> Result<Vec<f32>> {
    let dims = [1usize, 3, TARGET_H as usize, TARGET_W as usize];

    ctx.set_input(0, TensorType::F32, &dims, input)
        .map_err(|e| anyhow!("set_input failed: {:?}", e))?;

    ctx.compute()
        .map_err(|e| anyhow!("compute failed: {:?}", e))?;

    let mut output = vec![0f32; 200_000];
    let bytes = ctx
        .get_output(0, &mut output)
        .map_err(|e| anyhow!("get_output failed: {:?}", e))?;

    let elements = bytes / std::mem::size_of::<f32>();
    output.truncate(elements);
    Ok(output)
}

fn to_detections(output: &[f32]) -> Vec<Det> {
    let score = output.first().copied().unwrap_or(0.5);
    vec![Det {
        class: "car".to_string(),
        score,
        bbox: [0.1, 0.2, 0.4, 0.3],
    }]
}

#[no_mangle]
pub extern "C" fn _start() {
    if let Err(e) = run_loop() {
        eprintln!("guest error: {:?}", e);
    }
}

fn run_loop() -> Result<()> {
    let graph = load_graph("cubercnn_Res34_FPN_cuda.pt").context("load graph")?;
    let mut ctx = graph
        .init_execution_context()
        .map_err(|e| anyhow!("init_execution_context failed: {:?}", e))?;

    loop {
        let img_bytes = match read_exact_len() {
            Ok(b) => b,
            Err(e) => return Err(e),
        };

        let input = img_to_nchw(&img_bytes)?;
        let output = run_inference(&mut ctx, &input)?;
        let dets = to_detections(&output);

        let json = serde_json::to_value(&dets)?;
        write_json_result(&json)?;
    }
}
