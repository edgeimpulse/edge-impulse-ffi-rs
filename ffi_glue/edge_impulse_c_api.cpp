// C wrapper for Edge Impulse SDK FFI
#include "edge_impulse_wrapper.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"

// Forward declaration of the default impulse (C++ linkage)
extern ei_impulse_handle_t& ei_default_impulse;

extern "C" {

__attribute__((visibility("default"))) void ei_ffi_run_classifier_init(void) {
    ::run_classifier_init();
}

__attribute__((visibility("default"))) void ei_ffi_run_classifier_deinit(void) {
    ::run_classifier_deinit();
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_init_impulse(ei_impulse_handle_t* handle) {
    return ::init_impulse(handle);
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_run_classifier(signal_t* signal, ei_impulse_result_t* result, int debug) {
    return ::run_classifier(signal, result, debug);
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_run_classifier_continuous(signal_t* signal, ei_impulse_result_t* result, int debug, int enable_maf_unused) {
    return ::run_classifier_continuous(signal, result, debug, enable_maf_unused);
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_run_inference(ei_impulse_handle_t* handle, ei_feature_t* fmatrix, ei_impulse_result_t* result, int debug) {
    return ::run_inference(handle, fmatrix, result, debug);
}

// Helper function to create signal from buffer (like EIM binary)
__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_signal_from_buffer(const float* data, size_t data_size, signal_t* signal) {
    return static_cast<EI_IMPULSE_ERROR>(ei::numpy::signal_from_buffer(data, data_size, signal));
}

// Threshold setting functions
__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_set_object_detection_threshold(uint32_t block_id, float min_score) {
    // Find the learning block with the specified block_id
    for (size_t i = 0; i < ei_default_impulse.impulse->learning_blocks_size; i++) {
        const ei_learning_block_t& block = ei_default_impulse.impulse->learning_blocks[i];
        if (block.blockId == block_id) {
            // Check if this is a TensorFlow Lite graph block with object detection
            if (block.config != nullptr) {
                ei_learning_block_config_tflite_graph_t* config =
                    static_cast<ei_learning_block_config_tflite_graph_t*>(block.config);

                if (config->object_detection) {
                    config->threshold = min_score;
                    return EI_IMPULSE_OK;
                }
            }
        }
    }
    return EI_IMPULSE_INFERENCE_ERROR;
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_set_anomaly_threshold(uint32_t block_id, float min_anomaly_score) {
    // Find the learning block with the specified block_id
    for (size_t i = 0; i < ei_default_impulse.impulse->learning_blocks_size; i++) {
        const ei_learning_block_t& block = ei_default_impulse.impulse->learning_blocks[i];
        if (block.blockId == block_id) {
            // Check if this is an anomaly GMM block
            if (block.config != nullptr) {
                ei_learning_block_config_anomaly_gmm_t* config =
                    static_cast<ei_learning_block_config_anomaly_gmm_t*>(block.config);

                config->anomaly_threshold = min_anomaly_score;
                return EI_IMPULSE_OK;
            }
        }
    }
    return EI_IMPULSE_INFERENCE_ERROR;
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_set_object_tracking_threshold(uint32_t block_id, float threshold, uint32_t keep_grace, uint16_t max_observations) {
    // Find the postprocessing block with the specified block_id
    for (size_t i = 0; i < ei_default_impulse.impulse->postprocessing_blocks_size; i++) {
        const ei_postprocessing_block_t& block = ei_default_impulse.impulse->postprocessing_blocks[i];
        if (block.block_id == block_id) {
            // Check if this is an object tracking block
            if (block.config != nullptr) {
                ei_object_tracking_config_t* config =
                    static_cast<ei_object_tracking_config_t*>(block.config);

                config->threshold = threshold;
                config->keep_grace = keep_grace;
                config->max_observations = max_observations;
                return EI_IMPULSE_OK;
            }
        }
    }
    return EI_IMPULSE_INFERENCE_ERROR;
}

} // extern "C"
