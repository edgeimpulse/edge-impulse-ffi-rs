# FFI Glue Code for Edge Impulse Rust Bindings

This folder contains the C/C++ FFI glue code needed to bridge Edge Impulse C++ SDKs with Rust bindings.

At build time, these files will be copied into the model folder so you can swap in any Edge Impulse model code without losing your FFI logic.
