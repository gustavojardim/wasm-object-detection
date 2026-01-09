wit_bindgen::generate!({
    world: "ml-service",
    path: "wit"
});

fn main() {
    // Try to access the generated bindings
    println!("Bindings generated successfully");
    
    // Let's see what's available
    use wasi::nn::graph;
    use wasi::nn::tensor;
    use wasi::nn::inference;
    
    println!("All imports accessible!");
}
