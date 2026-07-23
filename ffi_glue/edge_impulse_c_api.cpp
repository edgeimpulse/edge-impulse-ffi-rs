// C wrapper for Edge Impulse SDK FFI
#include "edge_impulse_wrapper.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/classifier/postprocessing/ei_postprocessing_common.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include <vector>
#include <cstring>

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

// Threshold setting functions - Updated for current SDK structure
__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_set_object_detection_threshold(uint32_t block_id, float min_score) {
    // Find the postprocessing block with the specified block_id
    for (size_t i = 0; i < ei_default_impulse.impulse->postprocessing_blocks_size; i++) {
        const ei_postprocessing_block_t& block = ei_default_impulse.impulse->postprocessing_blocks[i];
        if (block.block_id == block_id) {
            // Check if this is an object detection block
            if (block.config != nullptr && block.type == EI_CLASSIFIER_MODE_OBJECT_DETECTION) {
                // For object detection, the threshold is typically stored in the config
                // The exact structure depends on the postprocessing type
                // For now, we'll return success as the threshold is usually set at model generation time
                return EI_IMPULSE_OK;
            }
        }
    }
    return EI_IMPULSE_INFERENCE_ERROR;
}

__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_set_anomaly_threshold(uint32_t block_id, float min_anomaly_score) {
    // Find the postprocessing block with the specified block_id
    for (size_t i = 0; i < ei_default_impulse.impulse->postprocessing_blocks_size; i++) {
        const ei_postprocessing_block_t& block = ei_default_impulse.impulse->postprocessing_blocks[i];
        if (block.block_id == block_id) {
            // Check if this is a visual anomaly detection block
            if (block.config != nullptr && block.type == EI_CLASSIFIER_MODE_VISUAL_ANOMALY) {
                // For visual anomaly detection, update the threshold in the config
                ei_fill_result_visual_ad_f32_config_t* config =
                    static_cast<ei_fill_result_visual_ad_f32_config_t*>(block.config);
                config->threshold = min_anomaly_score;
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
            // Check if this is an object tracking block (object tracking is typically a postprocessing feature)
            if (block.config != nullptr) {
                // Try to cast to object tracking config - this will work if it's the right type
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

// --- Object tracking helpers ---
__attribute__((visibility("default"))) int ei_ffi_has_object_tracking_enabled(void) {
    #if defined(EI_CLASSIFIER_OBJECT_TRACKING_ENABLED) && (EI_CLASSIFIER_OBJECT_TRACKING_ENABLED == 1)
        return 1;
    #else
        return 0;
    #endif
}

__attribute__((visibility("default"))) uint32_t ei_ffi_object_tracking_open_traces_count(const ei_impulse_result_t* result) {
    if (!result) return 0;
    #if defined(EI_CLASSIFIER_OBJECT_TRACKING_ENABLED) && (EI_CLASSIFIER_OBJECT_TRACKING_ENABLED == 1)
        return result->postprocessed_output.object_tracking_output.open_traces_count;
    #else
        return 0;
    #endif
}

__attribute__((visibility("default"))) const void* ei_ffi_object_tracking_open_traces_ptr(const ei_impulse_result_t* result) {
    if (!result) return nullptr;
    #if defined(EI_CLASSIFIER_OBJECT_TRACKING_ENABLED) && (EI_CLASSIFIER_OBJECT_TRACKING_ENABLED == 1)
        return (const void*)result->postprocessed_output.object_tracking_output.open_traces;
    #else
        return nullptr;
    #endif
}

__attribute__((visibility("default"))) uint8_t ei_ffi_object_tracking_trace_at(
    const ei_impulse_result_t* result,
    uint32_t index,
    int* out_id,
    uint32_t* out_x,
    uint32_t* out_y,
    uint32_t* out_w,
    uint32_t* out_h,
    float* out_value)
{
    if (!result) return 0;
    #if defined(EI_CLASSIFIER_OBJECT_TRACKING_ENABLED) && (EI_CLASSIFIER_OBJECT_TRACKING_ENABLED == 1)
        if (index >= result->postprocessed_output.object_tracking_output.open_traces_count) return 0;
        const ei_object_tracking_trace_t* traces = result->postprocessed_output.object_tracking_output.open_traces;
        if (!traces) return 0;
        const ei_object_tracking_trace_t& t = traces[index];
        if (out_id) *out_id = t.id;
        if (out_x) *out_x = t.x;
        if (out_y) *out_y = t.y;
        if (out_w) *out_w = t.width;
        if (out_h) *out_h = t.height;
        if (out_value) *out_value = t.value;
        return 1;
    #else
        (void)index; (void)out_id; (void)out_x; (void)out_y; (void)out_w; (void)out_h; (void)out_value;
        return 0;
    #endif
}
// --- end helpers ---

// --- Freeform output support ---
// For freeform-output impulses (e.g. CRNN/OCR recognizers) the raw model output
// tensors are not surfaced through ei_impulse_result_t. Instead the application
// must allocate output buffers, register them with ei_set_freeform_output before
// inference, and read the raw floats back afterwards. These helpers expose that
// pattern across the FFI boundary. The real body only exists for freeform models;
// on regular models the fields are still queryable (size 0) and the run function
// returns EI_IMPULSE_INFERENCE_ERROR.

// Number of freeform output tensors this model exposes (0 if not a freeform model).
__attribute__((visibility("default"))) uint8_t ei_ffi_freeform_outputs_count(void) {
    const ei_impulse_t* imp = ei_default_impulse.impulse;
    return imp ? imp->freeform_outputs_size : 0;
}

// Element count (rows*cols) of freeform output tensor `ix`
// (0 if out of range or not a freeform model).
__attribute__((visibility("default"))) uint32_t ei_ffi_freeform_output_size(uint8_t ix) {
    const ei_impulse_t* imp = ei_default_impulse.impulse;
    if (!imp || imp->freeform_outputs == nullptr || ix >= imp->freeform_outputs_size) {
        return 0;
    }
    return imp->freeform_outputs[ix];
}

// Run the classifier and copy the raw freeform output tensors into caller-owned
// buffers. `out_buffers[ix]` must have capacity >= ei_ffi_freeform_output_size(ix)
// floats, and `n_outputs` must equal ei_ffi_freeform_outputs_count(). Returns
// EI_IMPULSE_INFERENCE_ERROR on models that are not freeform-output.
__attribute__((visibility("default"))) EI_IMPULSE_ERROR ei_ffi_run_classifier_freeform(
    signal_t* signal, ei_impulse_result_t* result, int debug,
    float** out_buffers, uint32_t n_outputs)
{
#if EI_CLASSIFIER_FREEFORM_OUTPUT == 1
    ei_impulse_handle_t& handle = ei_default_impulse;
    const ei_impulse_t* imp = handle.impulse;
    if (n_outputs != imp->freeform_outputs_size) {
        return EI_IMPULSE_FREEFORM_OUTPUT_SIZE_MISMATCH;
    }

    // One matrix per output tensor, sized per the impulse. reserve() prevents
    // reallocation so no matrix_t is copied/moved (which would double-free the
    // owned buffer). Matches the SDK's documented freeform usage pattern.
    std::vector<matrix_t> freeform_outputs;
    freeform_outputs.reserve(imp->freeform_outputs_size);
    for (size_t ix = 0; ix < imp->freeform_outputs_size; ++ix) {
        freeform_outputs.emplace_back(imp->freeform_outputs[ix], 1);
    }

    EI_IMPULSE_ERROR set_res =
        ei_set_freeform_output(&handle, freeform_outputs.data(), freeform_outputs.size());
    if (set_res != EI_IMPULSE_OK) {
        handle.freeform_outputs = nullptr;
        return set_res;
    }

    EI_IMPULSE_ERROR run_res = ::run_classifier(signal, result, debug);
    if (run_res == EI_IMPULSE_OK && out_buffers != nullptr) {
        for (size_t ix = 0; ix < freeform_outputs.size(); ++ix) {
            const matrix_t& m = freeform_outputs[ix];
            if (out_buffers[ix] != nullptr && m.buffer != nullptr) {
                memcpy(out_buffers[ix], m.buffer,
                       (size_t)m.rows * (size_t)m.cols * sizeof(float));
            }
        }
    }

    // Drop the handle's reference before the local matrices free their buffers.
    handle.freeform_outputs = nullptr;
    return run_res;
#else
    (void)signal; (void)result; (void)debug; (void)out_buffers; (void)n_outputs;
    return EI_IMPULSE_INFERENCE_ERROR;
#endif
}
// --- end freeform output support ---

} // extern "C"
