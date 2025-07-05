use std::fs;
use std::path::Path;

/// Copy FFI glue files from ffi_glue/ to the selected model folder (e.g., cpp/ or cpp2/)
pub fn copy_ffi_glue(model_dir: &str) {
    let files = [
        "edge_impulse_c_api.cpp",
        "edge_impulse_wrapper.h",
        "CMakeLists.txt",
        "tflite_detection_postprocess_wrapper.cc",
    ];
    for file in &files {
        let src = format!("ffi_glue/{}", file);
        let dst = format!("{}/{}", model_dir, file);
        if Path::new(&src).exists() {
            fs::copy(&src, &dst).expect(&format!("Failed to copy {} to {}", src, dst));
        }
    }
}
