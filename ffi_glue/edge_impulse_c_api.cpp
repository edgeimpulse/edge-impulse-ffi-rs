// C wrapper for Edge Impulse SDK FFI
#include "edge_impulse_wrapper.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"

// Forward declaration of the default impulse (C++ linkage)
extern ei_impulse_handle_t& ei_default_impulse;

extern "C" {

void ei_ffi_run_classifier_init(void) {
    ::run_classifier_init();
}

void ei_ffi_run_classifier_deinit(void) {
    ::run_classifier_deinit();
}

EI_IMPULSE_ERROR ei_ffi_init_impulse(ei_impulse_handle_t* handle) {
    return ::init_impulse(handle);
}

EI_IMPULSE_ERROR ei_ffi_run_classifier(signal_t* signal, ei_impulse_result_t* result, int debug) {
    return ::run_classifier(signal, result, debug);
}

EI_IMPULSE_ERROR ei_ffi_run_classifier_continuous(signal_t* signal, ei_impulse_result_t* result, int debug, int enable_maf_unused) {
    return ::run_classifier_continuous(signal, result, debug, enable_maf_unused);
}

EI_IMPULSE_ERROR ei_ffi_run_inference(ei_impulse_handle_t* handle, ei_feature_t* fmatrix, ei_impulse_result_t* result, int debug) {
    return ::run_inference(handle, fmatrix, result, debug);
}

// Helper function to create signal from buffer (like EIM binary)
EI_IMPULSE_ERROR ei_ffi_signal_from_buffer(const float* data, size_t data_size, signal_t* signal) {
    return static_cast<EI_IMPULSE_ERROR>(ei::numpy::signal_from_buffer(data, data_size, signal));
}

} // extern "C"
