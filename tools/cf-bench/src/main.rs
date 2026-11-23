//! cf-bench <in.f32> <out.f32> <t>: four RGGB planes (t x t, f32 LE, white = 1) -> built-in denoiser -> RGB 2t x 2t.
use lcc_denoise_core::builtin::{Builtin, estimate_noise};
use lcc_denoise_core::TileRunner;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t: usize = a[3].parse().unwrap();
    let bytes = std::fs::read(&a[1]).unwrap();
    let input: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(input.len(), 4 * t * t);
    // the mosaic, as the app measures the photo's noise from it
    let s = 2 * t;
    let mut m = vec![0f32; s * s];
    for y in 0..t {
        for x in 0..t {
            let i = y * t + x;
            m[2 * y * s + 2 * x] = input[i];
            m[2 * y * s + 2 * x + 1] = input[t * t + i];
            m[(2 * y + 1) * s + 2 * x] = input[2 * t * t + i];
            m[(2 * y + 1) * s + 2 * x + 1] = input[3 * t * t + i];
        }
    }
    let noise = estimate_noise(&m, s, s);
    let t0 = std::time::Instant::now();
    let out = Builtin::with_noise(noise).run(&input).unwrap();
    eprintln!("noise a={:.3e} b={:.3e} time {:.2}s", noise.0, noise.1, t0.elapsed().as_secs_f64());
    let mut o = Vec::with_capacity(out.len() * 4);
    for v in out { o.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(&a[2], o).unwrap();
}
