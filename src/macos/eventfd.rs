// SPDX-License-Identifier: BSD-3-Clause
//
// macOS emulation of Linux `eventfd`.
//
// `eventfd(2)` is a Linux-specific kernel object: one fd, with
// `read(2)` returning a `u64` counter and `write(2)` adding to
// it, optionally non-blocking and optionally semaphore-mode.
// macOS has no native equivalent.
//
// For project-bifrost's use case (vhost-user kick/call doorbell
// wake-ups), we need:
//
//   - A `RawFd` that consumers (`AsRawFd`-trait users, pollers,
//     kqueue, `SCM_RIGHTS` recipients) can register and have
//     become readable when someone calls `write()` on the same
//     `EventFd`.
//   - `write(v)` that wakes any blocked reader.
//   - `read()` that returns the accumulated u64 counter and
//     clears it.
//   - `try_clone` that returns a separate `EventFd` referring
//     to the same kernel-side state.
//
// The emulation uses an internal AF_UNIX SOCK_STREAM
// socketpair. The two endpoints are kept inside the struct;
// `AsRawFd::as_raw_fd` returns the read end so anything polling
// the fd or sending it via `SCM_RIGHTS` sees the side that
// becomes readable when `write()` is called.  `write(v)` writes
// 8 bytes (little-endian u64) into the write end; `read()`
// drains all currently-readable 8-byte payloads from the read
// end and sums them — matching Linux eventfd's accumulator
// semantics.
//
// Why socketpair over `pipe(2)`:
//
//   - Linux eventfd is bidirectional ("either side can write,
//     either side can read"). SOCK_STREAM socketpair matches
//     that contract; `pipe(2)` does not.
//   - `EventFd::try_clone` semantics work cleanly: dup'ing
//     either end of the socketpair gives a fully-usable
//     bidirectional endpoint, while dup'ing a pipe end gives
//     only one direction.
//
// macOS does not honor `SOCK_NONBLOCK`/`SOCK_CLOEXEC` in
// socketpair's `type` argument (those are Linux extensions),
// so the flags are still applied via fcntl after creation.
//
// Why not `EVFILT_USER` or Mach ports:
//
//   - `EVFILT_USER` is the macOS-native equivalent of a
//     userspace wakeup, but kqueue fds do not survive
//     `SCM_RIGHTS` — they are per-process kernel state — so the
//     vhost-user kick/call path that ships the EventFd to a
//     peer process cannot use it.
//   - Mach ports are the truly native macOS IPC, but they do
//     not fit the `AsRawFd` API contract that vhost,
//     vhost-user-backend, and the wider rust-vmm ecosystem
//     build on. Wrapping a Mach port behind `AsRawFd` requires
//     a kqueue+EVFILT_MACHPORT bridge that recreates the
//     SCM_RIGHTS problem.
//
// Socketpair is the right layer of native-ness for the
// AsRawFd-based abstraction.
//
// Caveats vs. the Linux primitive:
//
// - Linux `eventfd` is one fd; we hold two internally. This means
//   when an `EventFd` is dropped, both ends close together. If a
//   peer received the read end via `SCM_RIGHTS`, the peer still
//   has its own read-end fd but loses the write end the *original*
//   owner had. This matches vhost-user kick/call: the peer never
//   needs to *write* to a kick fd, only read.
//
// - Linux semaphore-mode (`EFD_SEMAPHORE`) is not emulated; the
//   flag is ignored. None of the vhost-user code paths exercise
//   semaphore semantics.
//
// - `EFD_NONBLOCK` is honored; `EFD_CLOEXEC` is honored.

use std::io;
use std::mem;
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use std::result;

/// Make-the-pipe-nonblocking flag, mirroring Linux's `EFD_NONBLOCK`.
pub const EFD_NONBLOCK: i32 = 1;

/// Close-on-exec flag, mirroring Linux's `EFD_CLOEXEC`. Honored.
pub const EFD_CLOEXEC: i32 = 2;

/// Semaphore mode, mirroring Linux's `EFD_SEMAPHORE`. The flag is
/// accepted but the macOS emulation does not implement semaphore
/// semantics. Documented for source-compatibility only.
pub const EFD_SEMAPHORE: i32 = 4;

/// pipe-based eventfd emulation.
#[derive(Debug)]
pub struct EventFd {
    read_fd: RawFd,
    write_fd: RawFd,
}

impl EventFd {
    /// Create a new socketpair-backed eventfd.
    ///
    /// Uses `socketpair(AF_UNIX, SOCK_STREAM, …)` rather than
    /// `pipe(2)` because socketpair has **symmetric read/write
    /// semantics**, matching Linux eventfd's "either side can
    /// write, either side can read" contract. A pipe is
    /// strictly one-directional.
    ///
    /// macOS does not accept `SOCK_NONBLOCK`/`SOCK_CLOEXEC` in
    /// the `type` argument the way Linux does, so the flags are
    /// applied via fcntl after creation.
    ///
    /// The two ends are kept inside the struct; `AsRawFd`
    /// returns the read side so consumers (poll/kqueue/
    /// SCM_RIGHTS recipients) see the fd that becomes readable
    /// when `write()` is called. `EVFILT_USER` and Mach ports
    /// were considered as alternatives and rejected: the
    /// former cannot cross `SCM_RIGHTS` (per-process kqueue
    /// state), and the latter does not fit the `AsRawFd` API
    /// contract that vhost-user-backend builds on top of.
    pub fn new(flag: i32) -> result::Result<EventFd, io::Error> {
        let mut fds = [-1; 2];
        // SAFETY: fds is a valid two-element array.
        let rc = unsafe {
            libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr())
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        let efd = EventFd {
            read_fd: fds[0],
            write_fd: fds[1],
        };
        if flag & EFD_NONBLOCK != 0 {
            efd.set_nonblock(efd.read_fd)?;
            efd.set_nonblock(efd.write_fd)?;
        }
        if flag & EFD_CLOEXEC != 0 {
            efd.set_cloexec(efd.read_fd)?;
            efd.set_cloexec(efd.write_fd)?;
        }
        Ok(efd)
    }

    fn set_nonblock(&self, fd: RawFd) -> io::Result<()> {
        // SAFETY: fd is owned by this EventFd.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: same fd.
        let rc = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn set_cloexec(&self, fd: RawFd) -> io::Result<()> {
        // SAFETY: fd is owned by this EventFd.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: same fd.
        let rc = unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Increment the counter by `v`. Writes 8 bytes (little-endian)
    /// to the internal write end; the read end becomes readable.
    pub fn write(&self, v: u64) -> result::Result<(), io::Error> {
        let bytes = v.to_le_bytes();
        // SAFETY: write_fd is owned and bytes is a valid 8-byte buffer.
        let rc = unsafe {
            libc::write(
                self.write_fd,
                bytes.as_ptr() as *const libc::c_void,
                bytes.len(),
            )
        };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else if rc as usize != bytes.len() {
            // Partial write of an 8-byte payload is unexpected on a
            // pipe but defensively report it.
            Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short write to eventfd pipe",
            ))
        } else {
            Ok(())
        }
    }

    /// Drain all currently-readable counter increments and return
    /// their sum. Mirrors Linux `eventfd` read semantics: one call
    /// consumes the accumulated value.
    pub fn read(&self) -> result::Result<u64, io::Error> {
        let mut total: u64 = 0;
        let mut buf = [0u8; 8];
        loop {
            // SAFETY: read_fd is owned and buf is a valid 8-byte buffer.
            let rc = unsafe {
                libc::read(
                    self.read_fd,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf.len(),
                )
            };
            if rc == buf.len() as isize {
                total = total.wrapping_add(u64::from_le_bytes(buf));
                // Try to drain more, in case multiple writes have
                // accumulated. This matches Linux eventfd's behavior
                // of one read returning the sum of all queued writes.
                continue;
            } else if rc < 0 {
                let err = io::Error::last_os_error();
                if total == 0 {
                    return Err(err);
                }
                // EAGAIN on a non-blocking fd after at least one
                // successful read is the normal end-of-drain signal.
                return Ok(total);
            } else {
                // Partial read or EOF — return what we have.
                return Ok(total);
            }
        }
    }

    /// Duplicate both ends so the clone refers to the same pipe.
    pub fn try_clone(&self) -> result::Result<EventFd, io::Error> {
        // SAFETY: read_fd is owned and dup returns a fresh fd.
        let read_dup = unsafe { libc::dup(self.read_fd) };
        if read_dup < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: write_fd is owned and dup returns a fresh fd.
        let write_dup = unsafe { libc::dup(self.write_fd) };
        if write_dup < 0 {
            let err = io::Error::last_os_error();
            // SAFETY: read_dup was just dup'd, we own it.
            unsafe {
                libc::close(read_dup);
            }
            return Err(err);
        }
        Ok(EventFd {
            read_fd: read_dup,
            write_fd: write_dup,
        })
    }
}

impl AsRawFd for EventFd {
    /// The read end — the side that becomes readable when `write()`
    /// is called and the side that should be passed to `poll`,
    /// `kqueue`, or `SCM_RIGHTS`.
    fn as_raw_fd(&self) -> RawFd {
        self.read_fd
    }
}

impl FromRawFd for EventFd {
    /// Wraps an existing fd as an `EventFd`. The caller is asserting
    /// the fd already participates in a pipe-like pair; the second
    /// (write) end is left as a dup of the same fd so naïve
    /// single-fd consumers continue to work. This matches the
    /// existing rust-vmm API contract that
    /// `from_raw_fd(eventfd_fd)` produces a usable `EventFd`.
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        let write_dup = libc::dup(fd);
        EventFd {
            read_fd: fd,
            write_fd: if write_dup >= 0 { write_dup } else { fd },
        }
    }
}

impl IntoRawFd for EventFd {
    /// Surrender ownership of the read end. The write end is closed
    /// on the way out; callers consuming this fd take responsibility
    /// for half-eventfd semantics.
    fn into_raw_fd(self) -> RawFd {
        let read_fd = self.read_fd;
        let write_fd = self.write_fd;
        let _ = mem::ManuallyDrop::new(self);
        // SAFETY: write_fd is owned.
        if write_fd != read_fd {
            unsafe {
                libc::close(write_fd);
            }
        }
        read_fd
    }
}

impl Drop for EventFd {
    fn drop(&mut self) {
        // SAFETY: both fds owned.
        unsafe {
            libc::close(self.read_fd);
            if self.write_fd != self.read_fd {
                libc::close(self.write_fd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_returns_sum() {
        let efd = EventFd::new(EFD_NONBLOCK).expect("create");
        efd.write(3).expect("write 3");
        efd.write(5).expect("write 5");
        assert_eq!(efd.read().expect("read"), 8);
    }

    #[test]
    fn clone_shares_state() {
        let efd = EventFd::new(EFD_NONBLOCK).expect("create");
        let clone = efd.try_clone().expect("clone");
        efd.write(42).expect("write");
        // The clone reads from a dup'd read end of the same pipe,
        // so it sees the byte the original wrote.
        assert_eq!(clone.read().expect("read"), 42);
    }
}
