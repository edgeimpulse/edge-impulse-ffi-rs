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

#ifdef __cplusplus
}
#endif

#endif // EDGE_IMPULSE_WRAPPER_H
