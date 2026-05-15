// SPDX-License-Identifier: BSD-3-Clause
//
// macOS-specific shims for Linux-only primitives that the rust-vmm
// ecosystem assumes. These keep crates like `vhost` and
// `vhost-user-backend` compilable against
// `target_os = "macos"`, sufficient for project-bifrost's
// host-side vhost-user backend.
//
// Only the pieces that actually appear in dependency graphs are
// ported here; this is not a full Linux-API emulation layer.

/// pipe-based emulation of Linux `eventfd(2)` for macOS targets.
pub mod eventfd;
