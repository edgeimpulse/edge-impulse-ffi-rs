use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

// Add serde imports for JSON handling
use serde::Deserialize;
use std::path::Path;

// JSON response structures for Edge Impulse API
#[derive(Debug, Deserialize)]
struct ProjectResponse {
    success: bool,
    #[allow(dead_code)]
    project: Project,
    #[serde(rename = "defaultImpulseId")]
    default_impulse_id: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct Project {
    #[allow(dead_code)]
    id: i32,
    #[allow(dead_code)]
    name: String,
}

#[derive(Debug, Deserialize)]
struct BuildJobResponse {
    success: bool,
    id: i32,
}

#[derive(Debug, Deserialize)]
struct JobStatusResponse {
    success: bool,
    job: JobStatus,
}

#[derive(Debug, Deserialize)]
struct JobStatus {
    #[allow(dead_code)]
    id: i32,
    category: String,
    finished: Option<String>, // Can be a timestamp string when finished
    #[serde(rename = "finishedSuccessful")]
    finished_successful: Option<bool>,
}

/// Copy FFI glue files from ffi_glue/ to the selected model folder (e.g., cpp/ or cpp2/)
fn copy_ffi_glue(model_dir: &str) {
    let files = [
        "edge_impulse_c_api.cpp",
        "edge_impulse_wrapper.h",
        "CMakeLists.txt",
        "tflite_detection_postprocess_wrapper.cc",
    ];
    for file in &files {
        let src = format!("ffi_glue/{}", file);
        let dst = format!("{}/{}", model_dir, file);
        if std::path::Path::new(&src).exists() {
            fs::copy(&src, &dst).unwrap_or_else(|_| panic!("Failed to copy {} to {}", src, dst));
        }
    }
}

/// Copy model files from a custom directory specified by EI_MODEL environment variable
fn copy_model_from_custom_path() -> bool {
    if let Ok(model_path) = env::var("EI_MODEL") {
        println!(
            "cargo:info=Found EI_MODEL environment variable: {}",
            model_path
        );

        let model_source = Path::new(&model_path);
        if !model_source.exists() {
            println!("cargo:error=EI_MODEL path does not exist: {}", model_path);
            return false;
        }

        let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let model_dest = Path::new(&manifest_dir).join("model");

        // Create model directory if it doesn't exist
        if !model_dest.exists() {
            std::fs::create_dir_all(&model_dest)
                .unwrap_or_else(|_| panic!("Failed to create model directory"));
        }

        // Copy the model files
        println!(
            "cargo:info=Copying model files from {} to {}",
            model_path,
            model_dest.display()
        );

        // Copy directories that should exist in a valid model
        let dirs_to_copy = ["edge-impulse-sdk", "model-parameters", "tflite-model"];
        for dir in &dirs_to_copy {
            let src_dir = model_source.join(dir);
            let dst_dir = model_dest.join(dir);

            println!(
                "cargo:info=DEBUG: Checking source directory: {} (exists: {})",
                src_dir.display(),
                src_dir.exists()
            );

            if src_dir.exists() {
                if dst_dir.exists() {
                    std::fs::remove_dir_all(&dst_dir)
                        .unwrap_or_else(|_| panic!("Failed to remove existing {}", dir));
                }
                copy_dir_recursive(&src_dir, &dst_dir)
                    .unwrap_or_else(|_| panic!("Failed to copy {}", dir));
                println!("cargo:info=Copied {} directory", dir);

                // Debug: List contents after copy
                if dir == &"tflite-model" {
                    println!("cargo:info=DEBUG: Contents of copied tflite-model directory:");
                    match std::fs::read_dir(&dst_dir) {
                        Ok(entries) => {
                            for entry in entries {
                                match entry {
                                    Ok(entry) => {
                                        let file_name = entry.file_name();
                                        let file_name_str = file_name.to_string_lossy();
                                        let file_type = if entry.file_type().unwrap().is_dir() { "DIR" } else { "FILE" };
                                        println!("cargo:info=DEBUG:   {}: {}", file_type, file_name_str);
                                    }
                                    Err(e) => println!("cargo:warning=DEBUG: Failed to read copied directory entry: {}", e),
                                }
                            }
                        }
                        Err(e) => println!(
                            "cargo:warning=DEBUG: Failed to read copied tflite-model directory: {}",
                            e
                        ),
                    }
                }
            } else {
                println!(
                    "cargo:warning=Source directory {} not found in {}",
                    dir, model_path
                );
            }
        }

        // Also copy tensorflow-lite if it exists (for full TFLite builds)
        let tflite_src = model_source.join("tensorflow-lite");
        let tflite_dst = model_dest.join("tensorflow-lite");
        if tflite_src.exists() {
            if tflite_dst.exists() {
                std::fs::remove_dir_all(&tflite_dst)
                    .unwrap_or_else(|_| panic!("Failed to remove existing tensorflow-lite"));
            }
            copy_dir_recursive(&tflite_src, &tflite_dst)
                .unwrap_or_else(|_| panic!("Failed to copy tensorflow-lite"));
            println!("cargo:info=Copied tensorflow-lite directory");
        }

        return true;
    }
    false
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }

    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if ty.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Read Edge Impulse project configuration from environment variables
fn read_edge_impulse_config() -> Option<(String, String)> {
    // Check environment variables
    let env_project_id = std::env::var("EI_PROJECT_ID").ok();
    let env_api_key = std::env::var("EI_API_KEY").ok();
    if let (Some(pid), Some(key)) = (env_project_id, env_api_key) {
        return Some((pid, key));
    }

    // No configuration found
    None
}

/// Download Edge Impulse model from the REST API using curl
///
/// This function:
/// 1. Gets project information to find the default impulse ID
/// 2. Triggers a build job for the model
/// 3. Polls the job status until completion
/// 4. Downloads and extracts the model files
/// 5. Returns true if successful, false otherwise
fn download_model_from_edge_impulse(project_id: &str, api_key: &str) -> bool {
    println!("cargo:info=Starting model download process...");
    println!("cargo:info=Project ID: {}", project_id);
    println!(
        "cargo:info=API Key: {}...",
        &api_key[..api_key.len().min(8)]
    );

    // Get the Edge Impulse Studio host from environment or use default
    let studio_host = env::var("EDGE_IMPULSE_STUDIO_HOST")
        .unwrap_or_else(|_| "https://studio.edgeimpulse.com".to_string());

    let base_url = format!("{}/v1/api", studio_host);

    // Step 1: Get project information to find defaultImpulseId
    println!("cargo:info=Step 1/5: Getting project information...");
    let project_url = format!("{}/{}", base_url, project_id);

    let project_response: ProjectResponse =
        match ureq::get(&project_url).set("x-api-key", api_key).call() {
            Ok(response) => {
                if response.status() != 200 {
                    println!(
                        "cargo:error=Failed to get project info: HTTP {}",
                        response.status()
                    );
                    return false;
                }
                match response.into_json() {
                    Ok(data) => data,
                    Err(e) => {
                        println!("cargo:error=Failed to parse project response: {}", e);
                        return false;
                    }
                }
            }
            Err(e) => {
                println!("cargo:error=Failed to get project info: {}", e);
                return false;
            }
        };

    if !project_response.success {
        println!("cargo:error=Project API call was not successful");
        return false;
    }

    let default_impulse_id = match project_response.default_impulse_id {
        Some(id) => id,
        None => {
            println!("cargo:error=No default impulse ID found in project");
            return false;
        }
    };

    println!(
        "cargo:info=Found default impulse ID: {}",
        default_impulse_id
    );

    // Step 2: Trigger build job
    println!("cargo:info=Step 2/5: Triggering model build job...");
    let build_url = format!(
        "{}/{}/jobs/build-ondevice-model?type=zip&impulse={}",
        base_url, project_id, default_impulse_id
    );

    // Determine engine type from environment variable, default to tflite-eon
    let engine = env::var("EI_ENGINE").unwrap_or_else(|_| "tflite-eon".to_string());
    println!("cargo:info=Using engine: {}", engine);

    let build_response: BuildJobResponse = match ureq::post(&build_url)
        .set("x-api-key", api_key)
        .set("content-type", "application/json")
        .send_json(serde_json::json!({"engine": engine}))
    {
        Ok(response) => {
            if response.status() != 200 {
                println!(
                    "cargo:error=Failed to trigger build: HTTP {}",
                    response.status()
                );
                return false;
            }
            match response.into_json() {
                Ok(data) => data,
                Err(e) => {
                    println!("cargo:error=Failed to parse build response: {}", e);
                    return false;
                }
            }
        }
        Err(e) => {
            println!("cargo:error=Failed to trigger build: {}", e);
            return false;
        }
    };

    if !build_response.success {
        println!("cargo:error=Build job creation was not successful");
        return false;
    }

    let job_id = build_response.id;
    println!("cargo:info=Build job created with ID: {}", job_id);

    // Step 3: Poll job status until completion
    println!("cargo:info=Step 3/5: Waiting for model build to complete...");
    println!("cargo:info=This step typically takes 2-5 minutes. Polling every 5 seconds...");
    let status_url = format!("{}/{}/jobs/{}/status", base_url, project_id, job_id);

    let mut attempts = 0;
    const MAX_ATTEMPTS: u32 = 120; // 10 minutes with 5-second intervals

    loop {
        attempts += 1;
        if attempts > MAX_ATTEMPTS {
            println!(
                "cargo:error=Build timed out after {} attempts ({} minutes)",
                MAX_ATTEMPTS,
                MAX_ATTEMPTS * 5 / 60
            );
            println!("cargo:error=The model build is taking longer than expected. You can try again or check your Edge Impulse project.");
            return false;
        }

        // Wait 5 seconds between polls
        std::thread::sleep(Duration::from_secs(5));

        let status_response: JobStatusResponse =
            match ureq::get(&status_url).set("x-api-key", api_key).call() {
                Ok(response) => {
                    if response.status() != 200 {
                        println!(
                            "cargo:error=Failed to get job status: HTTP {}",
                            response.status()
                        );
                        return false;
                    }
                    match response.into_json() {
                        Ok(data) => data,
                        Err(e) => {
                            println!("cargo:error=Failed to parse job status: {}", e);
                            return false;
                        }
                    }
                }
                Err(e) => {
                    println!("cargo:error=Failed to get job status: {}", e);
                    return false;
                }
            };

        if !status_response.success {
            println!("cargo:error=Job status API call was not successful");
            return false;
        }

        let job = status_response.job;
        println!(
            "cargo:info=Build status: {} (polling attempt {}/{})",
            job.category, attempts, MAX_ATTEMPTS
        );

        // Check if job is finished
        if let Some(successful) = job.finished_successful {
            if job.finished.is_some() {
                if successful {
                    println!("cargo:info=Build completed successfully!");
                    break;
                } else {
                    println!("cargo:error=Build failed on Edge Impulse servers");
                    return false;
                }
            }
        }
    }

    // Step 4: Download the model
    println!("cargo:info=Step 4/5: Downloading built model...");
    let download_url = format!(
        "{}/{}/deployment/download?type=zip&impulse={}",
        base_url, project_id, default_impulse_id
    );

    // Create model directory if it doesn't exist
    let model_dir = PathBuf::from("model");
    if !model_dir.exists() {
        if let Err(e) = fs::create_dir(&model_dir) {
            println!("cargo:error=Failed to create model directory: {}", e);
            return false;
        }
    }

    // Download the model
    let download_response = match ureq::get(&download_url).set("x-api-key", api_key).call() {
        Ok(response) => {
            if response.status() != 200 {
                println!(
                    "cargo:error=Failed to download model: HTTP {}",
                    response.status()
                );
                return false;
            }
            response
        }
        Err(e) => {
            println!("cargo:error=Failed to download model: {}", e);
            return false;
        }
    };

    // Step 5: Extract the model
    println!("cargo:info=Step 5/5: Extracting model files...");

    // Read the ZIP data
    let mut zip_data = Vec::new();
    match download_response.into_reader().read_to_end(&mut zip_data) {
        Ok(_) => {}
        Err(e) => {
            println!("cargo:error=Failed to read download data: {}", e);
            return false;
        }
    }

    // Extract ZIP file
    let mut archive = match zip::ZipArchive::new(std::io::Cursor::new(zip_data)) {
        Ok(archive) => archive,
        Err(e) => {
            println!("cargo:error=Failed to read ZIP archive: {}", e);
            return false;
        }
    };

    // Preserve existing .gitignore and README.md if they exist
    let gitignore_content = fs::read_to_string(model_dir.join(".gitignore")).ok();
    let readme_content = fs::read_to_string(model_dir.join("README.md")).ok();

    // Extract all files
    for i in 0..archive.len() {
        let mut file = match archive.by_index(i) {
            Ok(file) => file,
            Err(e) => {
                println!("cargo:error=Failed to access file {} in ZIP: {}", i, e);
                continue;
            }
        };

        let file_path = match file.enclosed_name() {
            Some(path) => path,
            None => {
                println!(
                    "cargo:warning=Skipping file with invalid path: {}",
                    file.name()
                );
                continue;
            }
        };

        let target_path = model_dir.join(file_path);

        if file.name().ends_with('/') {
            // Create directory
            if let Err(e) = fs::create_dir_all(&target_path) {
                println!(
                    "cargo:error=Failed to create directory {:?}: {}",
                    target_path, e
                );
            }
        } else {
            // Create parent directory if it doesn't exist
            if let Some(parent) = target_path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    println!(
                        "cargo:error=Failed to create parent directory {:?}: {}",
                        parent, e
                    );
                    continue;
                }
            }

            // Extract file
            let mut target_file = match fs::File::create(&target_path) {
                Ok(file) => file,
                Err(e) => {
                    println!("cargo:error=Failed to create file {:?}: {}", target_path, e);
                    continue;
                }
            };

            if let Err(e) = std::io::copy(&mut file, &mut target_file) {
                println!("cargo:error=Failed to write file {:?}: {}", target_path, e);
            }
        }
    }

    // Restore .gitignore and README.md if they existed before
    if let Some(content) = gitignore_content {
        if let Err(e) = fs::write(model_dir.join(".gitignore"), content) {
            println!("cargo:warning=Failed to restore .gitignore: {}", e);
        }
    }
    if let Some(content) = readme_content {
        if let Err(e) = fs::write(model_dir.join("README.md"), content) {
            println!("cargo:warning=Failed to restore README.md: {}", e);
        }
    }

    println!("cargo:info=Model downloaded and extracted successfully!");
    println!("cargo:info=Model is now ready for use. Future builds will use the local copy.");

    true
}

fn clean_model_folder() {
    let model_dir = "model";

    // Check if model directory exists
    if fs::metadata(model_dir).is_err() {
        println!("Model directory does not exist, nothing to clean");
        return;
    }

    // Read all entries in the model directory
    let entries = match fs::read_dir(model_dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Failed to read model directory: {}", e);
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                eprintln!("Failed to read directory entry: {}", e);
                continue;
            }
        };

        let path = entry.path();
        let file_name = path.file_name().unwrap_or_default();

        // Skip README.md and .gitignore
        if file_name == "README.md" || file_name == ".gitignore" {
            continue;
        }

        // Remove the entry (file or directory)
        if path.is_dir() {
            if let Err(e) = fs::remove_dir_all(&path) {
                eprintln!("Failed to remove directory {:?}: {}", path, e);
            } else {
                println!("Removed directory: {:?}", path);
            }
        } else if let Err(e) = fs::remove_file(&path) {
            eprintln!("Failed to remove file {:?}: {}", path, e);
        } else {
            println!("Removed file: {:?}", path);
        }
    }

    println!("Model folder cleaned successfully. Only README.md and .gitignore remain.");
}

/// Fix the header file path in the generated header file to point to the correct TFLite file location
fn fix_header_file_path(build_dir: &Path) {
    let tflite_model_dir = build_dir.join("tflite-model");

    // Find the actual TFLite file (should be named tflite_learn_*.tflite)
    let tflite_files: Vec<_> = std::fs::read_dir(&tflite_model_dir)
        .expect("Failed to read tflite-model directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let file_name_os = entry.file_name();
            let file_name = file_name_os.to_str()?;
            if file_name.ends_with(".tflite") && file_name.starts_with("tflite_learn_") {
                Some((entry.path(), file_name.to_string()))
            } else {
                None
            }
        })
        .collect();

    if tflite_files.is_empty() {
        println!("cargo:warning=No tflite_learn_*.tflite file found in build directory");
        return;
    }

    // Fix all TFLite files and their corresponding headers
    for (tflite_file, tflite_filename) in &tflite_files {
        let base_name = tflite_filename.trim_end_matches(".tflite");
        let header_filename = format!("{}.h", base_name);
        let header_file = tflite_model_dir.join(&header_filename);

        if header_file.exists() && tflite_file.exists() {
            let content =
                std::fs::read_to_string(&header_file).expect("Failed to read header file");
            let abs_path = tflite_file
                .canonicalize()
                .expect("Failed to canonicalize tflite file path");
            let abs_path_str = abs_path.to_str().expect("Non-UTF8 path");

            // Create the INCBIN macro name from the base name
            let incbin_name = format!("incbin_{}", base_name);

            // Patch any INCBIN macro to use the absolute path
            let old_pattern = format!(
                "INCBIN({}, \"tflite-model/{}\");",
                incbin_name, tflite_filename
            );
            let new_pattern = format!("INCBIN({}, \"{}\");", incbin_name, abs_path_str);

            let fixed_content = content.replace(&old_pattern, &new_pattern);
            std::fs::write(&header_file, fixed_content).expect("Failed to write fixed header file");
            println!(
                "cargo:info=Fixed header file path in {} for {}",
                header_file.display(),
                tflite_filename
            );
        } else {
            println!(
                "cargo:warning=Header file or tflite file not found: {} or {}",
                header_file.display(),
                tflite_file.display()
            );
        }
    }
}

fn extract_and_write_model_metadata() {
    use std::collections::HashMap;
    use std::fs;
    let header_path = "model/model-parameters/model_metadata.h";
    let out_path = "src/model_metadata.rs";
    let header = fs::read_to_string(header_path).expect("Failed to read model_metadata.h");

    let mut out = String::from("// This file is @generated by build.rs. Do not edit manually.\n");
    out.push_str("// Model metadata constants extracted from model_metadata.h\n\n");

    let mut seen = HashMap::new();
    let mut raw_defs = Vec::new();

    // First pass: collect all constants and their raw values
    for line in header.lines() {
        if let Some(rest) = line.strip_prefix("#define ") {
            let mut parts = rest.splitn(3, ' ');
            let name = parts.next();
            let val = parts.next();
            let val_rest = parts.next();
            if let (Some(name), Some(val)) = (name, val) {
                let value = if let Some(rest) = val_rest {
                    // Value is everything after the name
                    format!("{} {}", val, rest).trim().to_string()
                } else {
                    val.trim().to_string()
                };
                if !(name.starts_with("EI_CLASSIFIER_") || name.starts_with("EI_ANOMALY_TYPE_")) {
                    continue;
                }
                if seen.contains_key(name) {
                    continue;
                }
                raw_defs.push((name.to_string(), value.clone()));
                seen.insert(name.to_string(), value);
            }
        }
    }

    // Helper to resolve a constant recursively
    fn resolve(name: &str, emitted: &HashMap<String, String>) -> Option<String> {
        let mut current = name;
        let mut count = 0;
        while let Some(val) = emitted.get(current) {
            if val.starts_with('"') && val.ends_with('"') {
                return Some(val.clone());
            }
            if val.parse::<i32>().is_ok() {
                return Some(val.clone());
            }
            if val.parse::<f32>().is_ok() {
                return Some(val.clone());
            }
            current = val;
            count += 1;
            if count > 10 {
                break;
            } // prevent infinite loop
        }
        None
    }

    // Second pass: resolve references and emit Rust constants
    let mut emitted = HashMap::new();
    for (name, val) in &raw_defs {
        // Omit type alias constants
        if val == "uint8_t" || val == "bool" || val == "size_t" {
            continue;
        }
        // Special-case EI_CLASSIFIER_SENSOR: always emit as i32
        if name == "EI_CLASSIFIER_SENSOR" {
            // Try to resolve recursively
            let resolved = if let Some(resolved) = resolve(val, &emitted) {
                resolved
            } else {
                val.clone()
            };
            if let Ok(num) = resolved.parse::<i32>() {
                out.push_str(&format!(
                    "pub const {}: i32 = {};
",
                    name, num
                ));
                emitted.insert(name.clone(), num.to_string());
            } else {
                out.push_str(&format!(
                    "// Could not resolve EI_CLASSIFIER_SENSOR as i32: {}\n",
                    resolved
                ));
            }
            continue;
        }
        // String constants
        if val.starts_with('"') && val.ends_with('"') {
            out.push_str(&format!(
                "pub const {}: &str = {};
",
                name, val
            ));
            emitted.insert(name.clone(), val.clone());
            continue;
        }
        // Numeric constants
        if let Ok(num) = val.parse::<i32>() {
            if num < 0 {
                out.push_str(&format!(
                    "pub const {}: i32 = {};
",
                    name, num
                ));
            } else {
                out.push_str(&format!(
                    "pub const {}: usize = {};
",
                    name, num
                ));
            }
            emitted.insert(name.clone(), val.clone());
            continue;
        }
        if let Ok(num) = val.parse::<f32>() {
            out.push_str(&format!(
                "pub const {}: f32 = {};
",
                name, num
            ));
            emitted.insert(name.clone(), val.clone());
            continue;
        }
        // Reference to another constant
        if let Some(resolved) = emitted.get(val) {
            // Use the resolved value and type
            if resolved.starts_with('"') && resolved.ends_with('"') {
                out.push_str(&format!(
                    "pub const {}: &str = {};
",
                    name, resolved
                ));
            } else if let Ok(num) = resolved.parse::<i32>() {
                if num < 0 {
                    out.push_str(&format!(
                        "pub const {}: i32 = {};
",
                        name, num
                    ));
                } else {
                    out.push_str(&format!(
                        "pub const {}: usize = {};
",
                        name, num
                    ));
                }
            } else if let Ok(num) = resolved.parse::<f32>() {
                out.push_str(&format!(
                    "pub const {}: f32 = {};
",
                    name, num
                ));
            } else {
                out.push_str(&format!(
                    "pub const {}: usize = {};
",
                    name, resolved
                ));
            }
            emitted.insert(name.clone(), resolved.clone());
            continue;
        }
        // Special case: EI_CLASSIFIER_SLICE_SIZE
        if name == "EI_CLASSIFIER_SLICE_SIZE" {
            // Try to resolve from other constants
            let raw_sample_count = emitted
                .get("EI_CLASSIFIER_RAW_SAMPLE_COUNT")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let slices_per_window = emitted
                .get("EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            let value = raw_sample_count / slices_per_window;
            out.push_str(&format!(
                "pub const EI_CLASSIFIER_SLICE_SIZE: usize = {};
",
                value
            ));
            emitted.insert(name.clone(), value.to_string());
            continue;
        }
        // Special case: EI_CLASSIFIER_RESIZE_MODE
        if name == "EI_CLASSIFIER_RESIZE_MODE" {
            // This should resolve to EI_CLASSIFIER_RESIZE_SQUASH (3)
            out.push_str(
                "pub const EI_CLASSIFIER_RESIZE_MODE: usize = EI_CLASSIFIER_RESIZE_SQUASH;\n",
            );
            emitted.insert(name.clone(), "EI_CLASSIFIER_RESIZE_SQUASH".to_string());
            continue;
        }
        // Fallback: emit as a reference (may cause build error, but better than omitting)
        out.push_str(&format!("// Could not resolve: {} = {}\n", name, val));
    }

    // Add missing constants that are referenced but not defined in the header
    if !emitted.contains_key("EI_CLASSIFIER_RESIZE_SQUASH") {
        out.push_str("pub const EI_CLASSIFIER_RESIZE_SQUASH: usize = 3;\n");
    }
    if !emitted.contains_key("EI_CLASSIFIER_RESIZE_FIT_SHORTEST") {
        out.push_str("pub const EI_CLASSIFIER_RESIZE_FIT_SHORTEST: usize = 1;\n");
    }
    if !emitted.contains_key("EI_CLASSIFIER_RESIZE_FIT_LONGEST") {
        out.push_str("pub const EI_CLASSIFIER_RESIZE_FIT_LONGEST: usize = 2;\n");
    }
    if !emitted.contains_key("EI_CLASSIFIER_LAST_LAYER_YOLOV5") {
        out.push_str("pub const EI_CLASSIFIER_LAST_LAYER_YOLOV5: usize = 0;\n");
    }

    fs::write(out_path, out).expect("Failed to write model_metadata.rs");
}

/// Patch model metadata to always include visual anomaly detection fields
fn patch_model_metadata_for_visual_anomaly(model_dir: &Path) {
    let metadata_header = model_dir.join("model-parameters/model_metadata.h");
    if let Ok(content) = std::fs::read_to_string(&metadata_header) {
        // Replace EI_CLASSIFIER_HAS_VISUAL_ANOMALY definition to always be 1
        let patched = regex::Regex::new(r"#define EI_CLASSIFIER_HAS_VISUAL_ANOMALY\s+\d+")
            .unwrap()
            .replace(&content, "#define EI_CLASSIFIER_HAS_VISUAL_ANOMALY 1");

        if patched != content {
            std::fs::write(&metadata_header, patched.as_bytes())
                .expect("Failed to patch model_metadata.h");
            println!("cargo:info=Patched model_metadata.h to enable visual anomaly detection");
        }
    }
}

fn patch_model_for_full_tflite(model_dir: &Path, use_full_tflite: bool) {
    if !use_full_tflite {
        return;
    }
    // Patch ei_run_classifier.h to always include tflite_full.h when USE_FULL_TFLITE=1
    let classifier_header = model_dir.join("edge-impulse-sdk/classifier/ei_run_classifier.h");
    if let Ok(content) = std::fs::read_to_string(&classifier_header) {
        let patched = regex::Regex::new(r#"(?s)(#if \(EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_TFLITE\) && \(EI_CLASSIFIER_COMPILED != 1\))(.+?)#elif EI_CLASSIFIER_COMPILED == 1"#)
            .unwrap()
            .replace(&content, |caps: &regex::Captures| {
                format!(
                    "{}\n#if defined(EI_CLASSIFIER_USE_FULL_TFLITE)\n#include \"edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h\"\n#else\n#include \"edge-impulse-sdk/classifier/inferencing_engines/tflite_micro.h\"\n#endif\n#elif EI_CLASSIFIER_COMPILED == 1",
                    &caps[1]
                )
            });
        std::fs::write(&classifier_header, patched.as_bytes())
            .expect("Failed to patch ei_run_classifier.h");
        println!("cargo:info=Patched ei_run_classifier.h for full TFLite");
    }
    // Patch model/CMakeLists.txt to filter out micro sources
    let cmake_lists = model_dir.join("CMakeLists.txt");
    if let Ok(content) = std::fs::read_to_string(&cmake_lists) {
        let patched = regex::Regex::new(r#"(# Find all model and SDK source files\nRECURSIVE_FIND_FILE_APPEND\(MODEL_SOURCE \"tflite-model\" \"\*\.cpp\"\)\nRECURSIVE_FIND_FILE_APPEND\(MODEL_SOURCE \"model-parameters\" \"\*\.cpp\"\)\nRECURSIVE_FIND_FILE_APPEND\(MODEL_SOURCE \"edge-impulse-sdk\" \"\*\.cpp\"\)\nRECURSIVE_FIND_FILE_APPEND\(MODEL_SOURCE \"edge-impulse-sdk/third_party\" \"\*\.cpp\"\))"#)
            .unwrap()
            .replace(&content, |_caps: &regex::Captures| {
                "# Find all model and SDK source files\nRECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE \"tflite-model\" \"*.cpp\")\nRECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE \"model-parameters\" \"*.cpp\")\n\n# Conditionally include Edge Impulse SDK source files\nif(EI_CLASSIFIER_USE_FULL_TFLITE)\n    RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE \"edge-impulse-sdk\" \"*.cpp\")\n    list(FILTER MODEL_SOURCE EXCLUDE REGEX \".*tensorflow/lite/micro.*\")\n    list(FILTER MODEL_SOURCE EXCLUDE REGEX \".*micro_interpreter.*\")\n    list(FILTER MODEL_SOURCE EXCLUDE REGEX \".*all_ops_resolver.*\")\nelse()\n    RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE \"edge-impulse-sdk\" \"*.cpp\")\nendif()\n\nRECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE \"edge-impulse-sdk/third_party\" \"*.cpp\")".to_string()
            });
        std::fs::write(&cmake_lists, patched.as_bytes())
            .expect("Failed to patch model/CMakeLists.txt");
        println!("cargo:info=Patched model/CMakeLists.txt for full TFLite");
    }
}

fn main() {
    println!("cargo:warning=DEBUG: Build script starting...");
    println!(
        "cargo:warning=DEBUG: Current directory: {:?}",
        std::env::current_dir().unwrap()
    );

    // Force rerun on every build
    println!("cargo:rerun-if-changed=build.rs");

    // Get the current working directory and construct absolute paths
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let manifest_path = PathBuf::from(manifest_dir);

    let model_header = manifest_path.join("model/model-parameters/model_metadata.h");
    let out_bindings = manifest_path.join("src/bindings.rs");
    let _out_metadata = manifest_path.join("src/model_metadata.rs");

    // Check if we have a valid model structure - only look for actual model components
    let sdk_dir = manifest_path.join("model/edge-impulse-sdk");
    let model_parameters_dir = manifest_path.join("model/model-parameters");
    let tflite_model_dir = manifest_path.join("model/tflite-model");

    // Check if we have the essential model components
    let mut has_valid_model =
        sdk_dir.exists() && model_parameters_dir.exists() && tflite_model_dir.exists();

    // If no valid model found, try to copy from EI_MODEL path first
    if !has_valid_model {
        println!("cargo:info=No valid model found locally, checking for EI_MODEL environment variable...");

        if copy_model_from_custom_path() {
            // Re-check if we now have a valid model
            has_valid_model =
                sdk_dir.exists() && model_parameters_dir.exists() && tflite_model_dir.exists();

            if has_valid_model {
                println!("cargo:info=Model was copied from EI_MODEL path successfully");
            } else {
                println!("cargo:warning=Model copy completed but model structure is still invalid");
            }
        } else {
            println!("cargo:info=No EI_MODEL environment variable found or copy failed");
        }
    }

    // If still no valid model found, try to download from Edge Impulse API
    if !has_valid_model {
        println!("cargo:info=No valid model found locally, checking for Edge Impulse API configuration...");

        if let Some((project_id, api_key)) = read_edge_impulse_config() {
            println!("cargo:info=Found Edge Impulse configuration in environment variables");

            // Attempt to download the model
            if download_model_from_edge_impulse(&project_id, &api_key) {
                // Re-check if we now have a valid model
                has_valid_model =
                    sdk_dir.exists() && model_parameters_dir.exists() && tflite_model_dir.exists();

                if has_valid_model {
                    println!(
                        "cargo:warning=Model was downloaded from Edge Impulse Studio project ID {}",
                        project_id
                    );
                } else {
                    println!("cargo:warning=Model download completed but model structure is still invalid");
                }
            } else {
                println!("cargo:warning=Failed to download model from Edge Impulse API");
            }
        } else {
            println!("cargo:info=No Edge Impulse configuration found in environment variables");
            println!("cargo:info=To enable automatic model download, set the following environment variables:");
            println!("cargo:info=EI_PROJECT_ID=your-project-id");
            println!("cargo:info=EI_API_KEY=your-api-key");
        }
    }

    // If we have a valid model, copy the FFI glue files to set up the build environment
    if has_valid_model {
        copy_ffi_glue("model");

        // Patch model metadata to always include visual anomaly detection fields
        patch_model_metadata_for_visual_anomaly(&manifest_path.join("model"));
    }

    if has_valid_model {
        println!("cargo:info=Valid Edge Impulse model found, generating real bindings...");

        // Generate real bindings using bindgen
        let wrapper_header = manifest_path.join("model/edge_impulse_wrapper.h");
        let bindings = bindgen::Builder::default()
            .header(wrapper_header.to_str().unwrap())
            .clang_arg("-xc++")
            .clang_arg("-std=c++17")
            .clang_arg("-Imodel")
            .clang_arg("-Imodel/edge-impulse-sdk")
            .clang_arg("-O3")
            .clang_arg("-flto")
            .clang_arg("-ffast-math")
            .clang_arg("-funroll-loops")
            // Force inclusion of visual anomaly detection fields for consistent bindings
            .clang_arg("-DEI_CLASSIFIER_HAS_VISUAL_ANOMALY=1")
            .rustified_enum(".*")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .prepend_enum_name(false)
            .translate_enum_integer_types(true)
            .derive_copy(true)
            .derive_debug(true)
            .derive_default(true)
            // Do NOT derive Eq, PartialEq, Hash, Ord, PartialOrd to avoid function pointer comparison warnings
            .derive_eq(false)
            .derive_hash(false)
            .derive_partialeq(false)
            .derive_partialord(false)
            .derive_ord(false)
            // Disable problematic traits for structs with function pointers
            .disable_name_namespacing()
            .disable_untagged_union()
            // Ignore INCBIN macro to avoid processing .tflite files
            .blocklist_item("INCBIN")
            .blocklist_item("incbin_.*")
            .blocklist_item("gincbin_.*")
            .allowlist_type("ei_impulse_handle_t")
            .allowlist_type("ei_impulse_result_t")
            .allowlist_type("ei_feature_t")
            .allowlist_type("ei_signal_t")
            .allowlist_type("EI_IMPULSE_ERROR")
            .allowlist_type("ei_impulse_result_classification_t")
            .allowlist_type("ei_impulse_result_bounding_box_t")
            .allowlist_type("ei_impulse_result_timing_t")
            .allowlist_type("ei_impulse_visual_ad_result_t")
            .allowlist_function("ei_ffi_run_classifier_init")
            .allowlist_function("ei_ffi_run_classifier_deinit")
            .allowlist_function("ei_ffi_init_impulse")
            .allowlist_function("ei_ffi_run_classifier")
            .allowlist_function("ei_ffi_run_classifier_continuous")
            .allowlist_function("ei_ffi_run_inference")
            .allowlist_function("ei_ffi_signal_from_buffer")
            .generate()
            .expect("Unable to generate bindings");

        bindings
            .write_to_file(&out_bindings)
            .expect("Couldn't write bindings!");

        // Add allow attributes to suppress warnings in generated bindings
        let bindings_content =
            std::fs::read_to_string(&out_bindings).expect("Failed to read generated bindings");
        let modified_content = format!(
            "#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]\n{}",
            bindings_content
        );
        std::fs::write(&out_bindings, modified_content).expect("Failed to write modified bindings");

        // Generate model metadata
        if model_header.exists() {
            extract_and_write_model_metadata();
        } else {
            println!("cargo:warning=Model metadata header not found, skipping metadata generation");
        }

        println!("cargo:info=Real bindings generated successfully!");
    } else {
        eprintln!("cargo:error=FFI crate requires a valid Edge Impulse model, but none was found");
        eprintln!("cargo:error=Please either:");
        eprintln!("cargo:error=  1. Ensure model files exist in the model/ directory");
        eprintln!("cargo:error=  2. Set EI_MODEL environment variable to copy from a custom path:");
        eprintln!("cargo:error=     export EI_MODEL=/path/to/your/model");
        eprintln!("cargo:error=  3. Set environment variables to download a model:");
        eprintln!("cargo:error=     export EI_PROJECT_ID=your-project-id");
        eprintln!("cargo:error=     export EI_API_KEY=your-api-key");
        std::process::exit(1);
    }

    // Check if we should clean the model folder
    if env::var("CLEAN_MODEL").is_ok() {
        clean_model_folder();
        return;
    }

    // Define model directory and build directory for use throughout the function
    let model_dir = "model";
    let cpp_dir = PathBuf::from(model_dir);
    let build_dir = cpp_dir.join("build");

    // If we have a valid model, we need to build the C++ library
    if has_valid_model {
        copy_ffi_glue(model_dir);

        // Create build directory if it doesn't exist
        std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

        // --- Dynamically find and copy TFLite file and header to build directory for INCBIN ---
        let tflite_model_dir = manifest_path.join("model/tflite-model");
        let tflite_build_dir = build_dir.join("tflite-model");

        // Debug: Print the tflite-model directory contents
        println!(
            "cargo:info=DEBUG: Checking tflite-model directory: {}",
            tflite_model_dir.display()
        );
        if !tflite_model_dir.exists() {
            println!("cargo:error=DEBUG: tflite-model directory does not exist!");
            std::process::exit(1);
        }

        // List all files in the directory
        match std::fs::read_dir(&tflite_model_dir) {
            Ok(entries) => {
                println!("cargo:info=DEBUG: Contents of tflite-model directory:");
                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let file_name = entry.file_name();
                            let file_name_str = file_name.to_string_lossy();
                            let file_type = if entry.file_type().unwrap().is_dir() {
                                "DIR"
                            } else {
                                "FILE"
                            };
                            println!("cargo:info=DEBUG:   {}: {}", file_type, file_name_str);
                        }
                        Err(e) => {
                            println!("cargo:warning=DEBUG: Failed to read directory entry: {}", e)
                        }
                    }
                }
            }
            Err(e) => {
                println!(
                    "cargo:error=DEBUG: Failed to read tflite-model directory: {}",
                    e
                );
                std::process::exit(1);
            }
        }

        // Find the actual TFLite file (should be named tflite_learn_*.tflite)
        let tflite_files: Vec<_> = std::fs::read_dir(&tflite_model_dir)
            .expect("Failed to read tflite-model directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let file_name_os = entry.file_name();
                let file_name = file_name_os.to_str()?;
                println!("cargo:info=DEBUG: Checking file: {} (ends_with .tflite: {}, starts_with tflite_learn_: {})",
                    file_name,
                    file_name.ends_with(".tflite"),
                    file_name.starts_with("tflite_learn_"));
                if file_name.ends_with(".tflite") && file_name.starts_with("tflite_learn_") {
                    Some((entry.path(), file_name.to_string()))
                } else {
                    None
                }
            })
            .collect();

        if tflite_files.is_empty() {
            println!("cargo:error=No tflite_learn_*.tflite file found in model/tflite-model/");
            std::process::exit(1);
        }

        std::fs::create_dir_all(&tflite_build_dir)
            .expect("Failed to create tflite-model build dir");

        // Copy all TFLite files and their corresponding headers
        for (tflite_source, tflite_filename) in &tflite_files {
            let base_name = tflite_filename.trim_end_matches(".tflite");
            let header_filename = format!("{}.h", base_name);
            let header_source = tflite_model_dir.join(&header_filename);

            if !header_source.exists() {
                println!(
                    "cargo:error=Header file {} not found for TFLite file {}",
                    header_filename, tflite_filename
                );
                std::process::exit(1);
            }

            let tflite_dest = tflite_build_dir.join(tflite_filename);
            let header_dest = tflite_build_dir.join(&header_filename);

            // Copy TFLite file
            if tflite_dest.exists() {
                std::fs::remove_file(&tflite_dest).expect("Failed to remove old TFLite file");
            }
            std::fs::copy(tflite_source, &tflite_dest)
                .expect("Failed to copy TFLite file to build directory");

            // Copy header file
            if header_dest.exists() {
                std::fs::remove_file(&header_dest).expect("Failed to remove old header file");
            }
            std::fs::copy(&header_source, &header_dest)
                .expect("Failed to copy header file to build directory");

            println!(
                "cargo:info=Copied TFLite files to build directory: {} -> {}",
                tflite_filename,
                tflite_dest.display()
            );
        }

        // Fix the header file paths in all copied header files
        fix_header_file_path(&build_dir);

        // Also overwrite the original headers to ensure C++ build uses the correct paths
        for (_, tflite_filename) in &tflite_files {
            let base_name = tflite_filename.trim_end_matches(".tflite");
            let header_filename = format!("{}.h", base_name);
            let header_source = tflite_model_dir.join(&header_filename);
            let header_dest = tflite_build_dir.join(&header_filename);
            std::fs::copy(&header_dest, &header_source)
                .expect("Failed to overwrite original header file with fixed path");
        }

        // Always remove the static library to force a rebuild if model or header changes
        let lib_path = build_dir.join("libedge-impulse-sdk.a");
        if lib_path.exists() {
            std::fs::remove_file(&lib_path).expect("Failed to remove old static library");
            println!("cargo:warning=Removed old static library to force C++ rebuild");
        }
    }
    // --- End TFLite copy logic ---

    // Check if we need full TensorFlow Lite
    // Only USE_FULL_TFLITE is supported
    let use_full_tflite = env::var("USE_FULL_TFLITE").is_ok();

    // Detect platform target
    let target_platform = if env::var("TARGET_MAC_ARM64").is_ok() {
        "mac-arm64"
    } else if env::var("TARGET_MAC_X86_64").is_ok() {
        "mac-x86_64"
    } else if env::var("TARGET_LINUX_X86").is_ok() {
        "linux-x86"
    } else if env::var("TARGET_LINUX_AARCH64").is_ok() {
        "linux-aarch64"
    } else if env::var("TARGET_LINUX_ARMV7").is_ok() {
        "linux-armv7"
    } else if env::var("TARGET_JETSON_NANO").is_ok() {
        "linux-jetson-nano"
    } else if env::var("TARGET_JETSON_ORIN").is_ok()
        || env::var("TARGET_RENESAS_RZV2L").is_ok()
        || env::var("TARGET_RENESAS_RZG2L").is_ok()
        || env::var("TARGET_AM68PA").is_ok()
        || env::var("TARGET_AM62A").is_ok()
        || env::var("TARGET_AM68A").is_ok()
        || env::var("TARGET_TDA4VM").is_ok()
    {
        "linux-aarch64"
    } else {
        // Auto-detect based on current system
        if cfg!(target_os = "macos") {
            if cfg!(target_arch = "aarch64") {
                "mac-arm64"
            } else {
                "mac-x86_64"
            }
        } else if cfg!(target_os = "linux") {
            if cfg!(target_arch = "aarch64") {
                "linux-aarch64"
            } else if cfg!(target_arch = "arm") {
                "linux-armv7"
            } else {
                "linux-x86"
            }
        } else {
            "linux-x86" // default fallback
        }
    };

    // Detect additional backend/accelerator support
    let use_tvm = env::var("USE_TVM").is_ok();
    let use_onnx = env::var("USE_ONNX").is_ok();
    let use_qualcomm_qnn = env::var("USE_QUALCOMM_QNN").is_ok();
    let use_ethos = env::var("USE_ETHOS").is_ok();
    let use_akida = env::var("USE_AKIDA").is_ok();
    let use_memryx = env::var("USE_MEMRYX").is_ok();
    let link_tflite_flex = env::var("LINK_TFLITE_FLEX_LIBRARY").is_ok();
    let use_memryx_software = env::var("EI_CLASSIFIER_USE_MEMRYX_SOFTWARE").is_ok();

    // Get TensorRT version for Jetson builds
    let tensorrt_version = env::var("TENSORRT_VERSION").unwrap_or_else(|_| "8.5.2".to_string());

    // Get Python cross path for cross-compilation
    let python_cross_path = env::var("PYTHON_CROSS_PATH").ok();

    // Configure CMake with the required macros for C linkage
    let mut cmake_args = vec![
        "..".to_string(),
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DEIDSP_SIGNAL_C_FN_POINTER=1".to_string(),
        "-DEI_C_LINKAGE=1".to_string(),
        "-DBUILD_SHARED_LIBS=OFF".to_string(), // Build static library
    ];

    if use_full_tflite {
        cmake_args.push("-DEI_CLASSIFIER_USE_FULL_TFLITE=1".to_string());
        cmake_args.push(format!("-DTARGET_PLATFORM={}", target_platform));
        println!(
            "cargo:info=Building with full TensorFlow Lite for platform: {}",
            target_platform
        );
    } else {
        println!("cargo:info=Building with TensorFlow Lite Micro");
    }

    // Pass additional backend/accelerator flags
    if use_tvm {
        cmake_args.push("-DUSE_TVM=1".to_string());
        println!("cargo:info=Building with Apache TVM support");
    }
    if use_onnx {
        cmake_args.push("-DUSE_ONNX=1".to_string());
        println!("cargo:info=Building with ONNX Runtime support");
    }
    if use_qualcomm_qnn {
        cmake_args.push("-DUSE_QUALCOMM_QNN=1".to_string());
        println!("cargo:info=Building with Qualcomm QNN support");
    }
    if use_ethos {
        cmake_args.push("-DUSE_ETHOS=1".to_string());
        println!("cargo:info=Building with ARM Ethos support");
    }
    if use_akida {
        cmake_args.push("-DUSE_AKIDA=1".to_string());
        println!("cargo:info=Building with BrainChip Akida support");
    }
    if use_memryx {
        cmake_args.push("-DUSE_MEMRYX=1".to_string());
        println!("cargo:info=Building with MemryX support");
    }
    if link_tflite_flex {
        cmake_args.push("-DLINK_TFLITE_FLEX_LIBRARY=1".to_string());
        println!("cargo:info=Linking TensorFlow Lite Flex library");
    }
    if use_memryx_software {
        cmake_args.push("-DEI_CLASSIFIER_USE_MEMRYX_SOFTWARE=1".to_string());
        println!("cargo:info=Using MemryX software mode");
    }

    // Pass TensorRT version for Jetson builds
    cmake_args.push(format!("-DTENSORRT_VERSION={}", tensorrt_version));

    // Pass Python cross path if specified
    if let Some(ref path) = python_cross_path {
        cmake_args.push(format!("-DPYTHON_CROSS_PATH={}", path));
    }

    // If we have a valid model, check if we need to build the C++ library
    if has_valid_model {
        // Check if the library already exists
        let lib_path = build_dir.join("libedge-impulse-sdk.a");
        let should_rebuild = !lib_path.exists() || env::var("FORCE_REBUILD").is_ok();

        if should_rebuild {
            if !lib_path.exists() {
                println!("cargo:warning=Library not found, building C++ library...");
            } else {
                println!("cargo:warning=Force rebuild requested, rebuilding C++ library...");
            }

            println!("cargo:warning=CMake args: {:?}", cmake_args);
            let cmake_status = Command::new("cmake")
                .args(&cmake_args)
                .current_dir(&build_dir)
                .status()
                .expect("Failed to run cmake configure");

            if !cmake_status.success() {
                panic!("CMake configuration failed");
            }

            // Build the library
            let make_status = Command::new("make")
                .arg("-j")
                .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "4".to_string()))
                .current_dir(&build_dir)
                .status()
                .expect("Failed to run make");

            if !make_status.success() {
                panic!("Make build failed");
            }
        } else {
            println!("cargo:warning=Library already exists, skipping build");
        }

        // Diagnostic: print contents of build directory
        let entries = std::fs::read_dir(&build_dir).expect("Failed to read build directory");
        println!("Contents of {}:", build_dir.display());
        for entry in entries {
            let entry = entry.expect("Failed to read entry");
            println!("  {}", entry.file_name().to_string_lossy());
        }
    }

    // If we have a valid model, always set up library linking (regardless of whether we built it or not)
    if has_valid_model {
        println!("cargo:info=Setting up library linking for valid model");
        println!("cargo:info=Build directory: {}", build_dir.display());

        // Tell Cargo where to find the built library - use absolute path
        let absolute_build_dir = build_dir
            .canonicalize()
            .expect("Failed to get absolute path");
        println!(
            "cargo:rustc-link-search=native={}",
            absolute_build_dir.display()
        );

        // Link against the Edge Impulse SDK library
        // The library name will depend on what CMake generates, typically something like "edge-impulse-sdk"
        println!("cargo:rustc-link-lib=static=edge-impulse-sdk");

        // Link against C++ standard library
        println!("cargo:rustc-link-lib=c++");

        // Link against prebuilt TensorFlow Lite libraries when using full TensorFlow Lite
        if use_full_tflite {
            let tflite_lib_dir = format!("tflite/{}", target_platform);
            let tflite_lib_path = Path::new(&tflite_lib_dir);
            let cwd = std::env::current_dir().unwrap();
            println!("cargo:warning=DEBUG: current_dir: {}", cwd.display());
            println!("cargo:warning=DEBUG: tflite_lib_dir: {}", tflite_lib_dir);
            println!(
                "cargo:warning=DEBUG: tflite_lib_path exists: {}",
                tflite_lib_path.exists()
            );
            println!(
                "cargo:warning=DEBUG: tflite_lib_path absolute: {}",
                tflite_lib_path
                    .canonicalize()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "(not found)".to_string())
            );
            // Check if TensorFlow Lite libraries exist (they might not when building from git)
            if tflite_lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", tflite_lib_dir);

                // Link against prebuilt TensorFlow Lite and XNNPACK libraries in the correct order
                // This matches the official Makefile: -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthreadpool
                println!("cargo:rustc-link-lib=static=tensorflow-lite");
                println!("cargo:rustc-link-lib=static=cpuinfo");
                println!("cargo:rustc-link-lib=static=farmhash");
                println!("cargo:rustc-link-lib=static=fft2d_fftsg");
                println!("cargo:rustc-link-lib=static=fft2d_fftsg2d");
                println!("cargo:rustc-link-lib=static=ruy");
                println!("cargo:rustc-link-lib=static=XNNPACK");
                println!("cargo:rustc-link-lib=static=pthreadpool");
                println!("cargo:rustc-link-lib=static=flatbuffers");

                // Add system libraries that TensorFlow Lite depends on
                println!("cargo:rustc-link-lib=dl");

                println!("cargo:info=Linked against prebuilt TensorFlow Lite libraries");
            } else {
                println!("cargo:warning=TensorFlow Lite libraries not found at {}, skipping prebuilt library linking", tflite_lib_dir);
                println!("cargo:warning=This is expected when building from git. The CMake build will handle TensorFlow Lite linking.");
            }
        }

        // Re-run if any of the source files change
        println!("cargo:rerun-if-changed={}/CMakeLists.txt", model_dir);
        println!(
            "cargo:rerun-if-changed={}/edge_impulse_wrapper.h",
            model_dir
        );
        println!("cargo:rerun-if-changed={}/edge-impulse-sdk", model_dir);
        println!("cargo:rerun-if-changed={}/model-parameters", model_dir);
        println!("cargo:rerun-if-changed={}/tflite-model", model_dir);

        // Watch all TFLite files and their corresponding headers/CPP files
        let tflite_model_dir = Path::new(model_dir).join("tflite-model");
        if let Ok(entries) = std::fs::read_dir(&tflite_model_dir) {
            for entry in entries.flatten() {
                let file_name_os = entry.file_name();
                let file_name = file_name_os.to_string_lossy();
                if file_name.ends_with(".tflite") && file_name.starts_with("tflite_learn_") {
                    let base_name = file_name.trim_end_matches(".tflite");
                    let header_file = format!("{}.h", base_name);
                    let cpp_file = format!("{}.cpp", base_name);

                    println!(
                        "cargo:rerun-if-changed={}/tflite-model/{}",
                        model_dir, header_file
                    );
                    println!(
                        "cargo:rerun-if-changed={}/tflite-model/{}",
                        model_dir, cpp_file
                    );
                }
            }
        }

        println!("cargo:info=Library linking setup complete");
    } else {
        println!("cargo:info=No valid model found, skipping library linking");
    }

    // Only extract model metadata if we have a valid model
    if has_valid_model {
        extract_and_write_model_metadata();
        // Emit cargo:root for dependents
        println!("cargo:root={}", build_dir.display());
    }

    // Call this function after model download/extract and before C++ build
    patch_model_for_full_tflite(&manifest_path.join("model"), use_full_tflite);
}
