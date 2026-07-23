#ifndef EDGE_IMPULSE_WRAPPER_H
#define EDGE_IMPULSE_WRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Force inclusion of visual anomaly detection fields
#define EI_CLASSIFIER_HAS_VISUAL_ANOMALY 1

// Include the SDK headers for type definitions
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/classifier/ei_classifier_types.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/dsp/numpy_types.h"


#ifdef __cplusplus
extern "C" {
#endif

// Function declarations (no type redefinitions!)
void ei_ffi_run_classifier_init(void);
void ei_ffi_run_classifier_deinit(void);
EI_IMPULSE_ERROR ei_ffi_init_impulse(ei_impulse_handle_t* handle);
EI_IMPULSE_ERROR ei_ffi_run_classifier(signal_t* signal, ei_impulse_result_t* result, int debug);
EI_IMPULSE_ERROR ei_ffi_run_classifier_continuous(signal_t* signal, ei_impulse_result_t* result, int debug, int enable_maf_unused);
EI_IMPULSE_ERROR ei_ffi_run_inference(ei_impulse_handle_t* handle, ei_feature_t* fmatrix, ei_impulse_result_t* result, int debug);
// Helper function to create signal from buffer (like EIM binary)
EI_IMPULSE_ERROR ei_ffi_signal_from_buffer(const float* data, size_t data_size, signal_t* signal);

// Threshold setting functions
EI_IMPULSE_ERROR ei_ffi_set_object_detection_threshold(uint32_t block_id, float min_score);
EI_IMPULSE_ERROR ei_ffi_set_anomaly_threshold(uint32_t block_id, float min_anomaly_score);
EI_IMPULSE_ERROR ei_ffi_set_object_tracking_threshold(uint32_t block_id, float threshold, uint32_t keep_grace, uint16_t max_observations);

// Object tracking helpers (safe defaults when tracking is disabled)
int ei_ffi_has_object_tracking_enabled(void);
uint32_t ei_ffi_object_tracking_open_traces_count(const ei_impulse_result_t* result);
const void* ei_ffi_object_tracking_open_traces_ptr(const ei_impulse_result_t* result);
// Get a trace by index, returns 1 on success, 0 otherwise. Outputs filled only on success.
uint8_t ei_ffi_object_tracking_trace_at(const ei_impulse_result_t* result,
    uint32_t index,
    int* out_id,
    uint32_t* out_x,
    uint32_t* out_y,
    uint32_t* out_w,
    uint32_t* out_h,
    float* out_value);

// Freeform output helpers (for CRNN/OCR-style models whose raw output tensors are
// not surfaced through ei_impulse_result_t). See edge_impulse_c_api.cpp.
uint8_t ei_ffi_freeform_outputs_count(void);
uint32_t ei_ffi_freeform_output_size(uint8_t ix);
EI_IMPULSE_ERROR ei_ffi_run_classifier_freeform(
    signal_t* signal, ei_impulse_result_t* result, int debug,
    float** out_buffers, uint32_t n_outputs);

#ifdef __cplusplus
}
#endif

#endif // EDGE_IMPULSE_WRAPPER_H
