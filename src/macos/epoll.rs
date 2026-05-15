// SPDX-License-Identifier: BSD-3-Clause
//
// macOS emulation of Linux `epoll(7)` on top of BSD `kqueue(2)`.
//
// Linux `epoll` and BSD `kqueue` solve the same problem
// (event-driven readiness multiplexing across many fds) with
// substantially different surfaces. This module exposes the
// rust-vmm `Epoll`/`EpollEvent`/`EventSet`/`ControlOperation`
// API on top of a kqueue file descriptor so that crates like
// `vhost-user-backend` compile and operate on macOS targets.
//
// Translation rules:
//
//   EventSet::IN    ↔  EVFILT_READ
//   EventSet::OUT   ↔  EVFILT_WRITE
//   EDGE_TRIGGERED  ↔  EV_CLEAR
//   ONE_SHOT        ↔  EV_ONESHOT
//   ControlOperation::Add     ↔  EV_ADD
//   ControlOperation::Modify  ↔  EV_ADD (kqueue updates in place)
//   ControlOperation::Delete  ↔  EV_DELETE
//
// epoll uses a single fd-keyed registration with one bitmask of
// event types. kqueue uses one kevent per (fd, filter) pair, so
// an Add for `IN|OUT` registers two kevents. The Delete path
// removes both filters and ignores ENOENT on whichever was not
// previously registered. user_data flows through kqueue's
// `udata` field (a `void *`-shaped slot we use to carry the u64
// caller-provided cookie).

#![allow(clippy::missing_safety_doc)]

use std::io;
use std::ops::Deref;
use std::os::unix::io::{AsRawFd, RawFd};

use bitflags::bitflags;
use libc::{
    c_int, c_void, kevent, kqueue, EVFILT_READ, EVFILT_WRITE, EV_ADD, EV_CLEAR, EV_DELETE,
    EV_ONESHOT, EV_RECEIPT,
};

/// Layout-compatible stand-in for Linux's `libc::epoll_event`.
/// `EpollEvent::Deref` returns a reference to one of these so
/// existing callers can field-access `events` and `u64`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MacosEpollEvent {
    /// Bits set from `EventSet`. Matches the semantic of
    /// `epoll_event.events` on Linux.
    pub events: u32,
    /// Caller-supplied cookie — typically an fd or a vring index.
    pub u64: u64,
}

impl std::fmt::Debug for MacosEpollEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MacosEpollEvent {{ events: {:#x}, u64: {} }}", self.events, self.u64)
    }
}

// Bit values for EventSet. We use the same numeric values that
// glibc assigns to EPOLLIN/EPOLLOUT/etc. so `EventSet::from_bits`
// is consistent across platforms when a Linux-borne u32 happens
// to round-trip through our code paths.
const E_IN: u32 = 0x001;
const E_PRI: u32 = 0x002;
const E_OUT: u32 = 0x004;
const E_ERR: u32 = 0x008;
const E_HUP: u32 = 0x010;
const E_RDHUP: u32 = 0x2000;
const E_EXCLUSIVE: u32 = 1 << 28;
const E_WAKEUP: u32 = 1 << 29;
const E_ONESHOT: u32 = 1 << 30;
const E_ET: u32 = 1 << 31;

bitflags! {
    /// Subset of Linux epoll's event-type bitmask exposed on
    /// macOS. Bit values match Linux's `EPOLL*` constants so a
    /// u32 from `from_bits` is portable.
    pub struct EventSet: u32 {
        /// fd is readable.
        const IN = E_IN;
        /// fd is writable.
        const OUT = E_OUT;
        /// Error condition.
        const ERROR = E_ERR;
        /// Peer shut down the read side.
        const READ_HANG_UP = E_RDHUP;
        /// Edge-triggered mode (kqueue EV_CLEAR).
        const EDGE_TRIGGERED = E_ET;
        /// Hang up.
        const HANG_UP = E_HUP;
        /// High-priority data.
        const PRIORITY = E_PRI;
        /// Suspends process during event delivery (unused on macOS).
        const WAKE_UP = E_WAKEUP;
        /// One-shot delivery (kqueue EV_ONESHOT).
        const ONE_SHOT = E_ONESHOT;
        /// Exclusive wake-up (unused on macOS — kqueue has no
        /// equivalent; flag accepted for source compat).
        const EXCLUSIVE = E_EXCLUSIVE;
    }
}

/// `EPOLL_CTL_*` operation enumeration.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum ControlOperation {
    /// Register a new fd.
    Add = 1,
    /// Update an existing registration. kqueue's `EV_ADD` is
    /// idempotent and merges, so Add and Modify map identically.
    Modify = 2,
    /// Remove a registration.
    Delete = 3,
}

/// `epoll_event`-shaped value carrying an event mask and user data.
#[derive(Clone, Copy)]
pub struct EpollEvent(MacosEpollEvent);

impl std::fmt::Debug for EpollEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ events: {}, data: {} }}", self.events(), self.data())
    }
}

impl Deref for EpollEvent {
    type Target = MacosEpollEvent;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Default for EpollEvent {
    fn default() -> Self {
        EpollEvent(MacosEpollEvent { events: 0, u64: 0 })
    }
}

impl EpollEvent {
    /// Construct from an event-type mask plus a user cookie.
    pub fn new(events: EventSet, data: u64) -> Self {
        EpollEvent(MacosEpollEvent {
            events: events.bits(),
            u64: data,
        })
    }

    /// Raw event-bit mask.
    pub fn events(&self) -> u32 {
        self.0.events
    }

    /// Decoded `EventSet`. Unknown bits are dropped.
    pub fn event_set(&self) -> EventSet {
        EventSet::from_bits_truncate(self.0.events)
    }

    /// User cookie.
    pub fn data(&self) -> u64 {
        self.0.u64
    }

    /// User cookie reinterpreted as a `RawFd`. Mirrors the common
    /// epoll-on-Linux idiom where the data slot carries the fd.
    pub fn fd(&self) -> RawFd {
        self.0.u64 as i32
    }
}

/// macOS `Epoll` — a kqueue file descriptor presenting an
/// epoll-shaped API.
#[derive(Debug)]
pub struct Epoll {
    kq: RawFd,
}

impl Epoll {
    /// Create a new kqueue.
    pub fn new() -> io::Result<Self> {
        // SAFETY: no preconditions; checked return.
        let kq = unsafe { kqueue() };
        if kq < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(Epoll { kq })
        }
    }

    /// Apply an epoll-style control operation, translating to
    /// kqueue kevent registrations under the hood.
    pub fn ctl(
        &self,
        operation: ControlOperation,
        fd: RawFd,
        event: EpollEvent,
    ) -> io::Result<()> {
        let evset = event.event_set();
        let mut changes: Vec<libc::kevent> = Vec::with_capacity(2);
        let flags_common = match operation {
            ControlOperation::Add | ControlOperation::Modify => {
                let mut f = EV_ADD | EV_RECEIPT;
                if evset.contains(EventSet::EDGE_TRIGGERED) {
                    f |= EV_CLEAR;
                }
                if evset.contains(EventSet::ONE_SHOT) {
                    f |= EV_ONESHOT;
                }
                f
            }
            ControlOperation::Delete => EV_DELETE | EV_RECEIPT,
        };

        let udata = event.data() as *mut c_void;

        if matches!(
            operation,
            ControlOperation::Add | ControlOperation::Modify | ControlOperation::Delete,
        ) {
            if evset.contains(EventSet::IN) || matches!(operation, ControlOperation::Delete) {
                changes.push(make_kevent(fd, EVFILT_READ, flags_common, udata));
            }
            if evset.contains(EventSet::OUT) || matches!(operation, ControlOperation::Delete) {
                changes.push(make_kevent(fd, EVFILT_WRITE, flags_common, udata));
            }
        }

        if changes.is_empty() {
            return Ok(());
        }

        // SAFETY: kq is owned, changes is a valid slice.
        let rc = unsafe {
            kevent(
                self.kq,
                changes.as_ptr(),
                changes.len() as c_int,
                changes.as_mut_ptr(),
                changes.len() as c_int,
                std::ptr::null(),
            )
        };
        if rc < 0 {
            return Err(io::Error::last_os_error());
        }

        // With EV_RECEIPT each input kevent produces a response
        // kevent carrying any per-change error in `data`. Surface
        // the first non-trivial error, ignoring ENOENT during
        // Delete because kqueue rejects deletes of unregistered
        // (fd, filter) pairs and epoll's Delete is whole-fd.
        for ev in &changes {
            if ev.flags & EV_ERROR as u16 != 0 && ev.data != 0 {
                let errno = ev.data as i32;
                if matches!(operation, ControlOperation::Delete) && errno == libc::ENOENT {
                    continue;
                }
                return Err(io::Error::from_raw_os_error(errno));
            }
        }
        Ok(())
    }

    /// Wait for events. `timeout` is in milliseconds; -1 blocks
    /// forever, 0 polls. Writes up to `events.len()` events into
    /// the slice and returns how many were filled.
    pub fn wait(&self, timeout: i32, events: &mut [EpollEvent]) -> io::Result<usize> {
        if events.is_empty() {
            return Ok(0);
        }
        let mut kevents: Vec<libc::kevent> = vec![empty_kevent(); events.len()];

        let ts = if timeout < 0 {
            None
        } else {
            Some(libc::timespec {
                tv_sec: (timeout / 1000) as libc::time_t,
                tv_nsec: ((timeout % 1000) * 1_000_000) as libc::c_long,
            })
        };
        let ts_ptr = ts.as_ref().map_or(std::ptr::null(), |t| t as *const _);

        // SAFETY: kq is owned, kevents is a valid out-buffer.
        let rc = unsafe {
            kevent(
                self.kq,
                std::ptr::null(),
                0,
                kevents.as_mut_ptr(),
                kevents.len() as c_int,
                ts_ptr,
            )
        };
        if rc < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                return Ok(0);
            }
            return Err(err);
        }

        for i in 0..rc as usize {
            let ev = &kevents[i];
            let mut bits: u32 = 0;
            match ev.filter as i32 {
                f if f == EVFILT_READ as i32 => bits |= E_IN,
                f if f == EVFILT_WRITE as i32 => bits |= E_OUT,
                _ => {}
            }
            if ev.flags & libc::EV_EOF != 0 {
                bits |= E_HUP;
            }
            if ev.flags & EV_ERROR as u16 != 0 && ev.data != 0 {
                bits |= E_ERR;
            }
            events[i] = EpollEvent(MacosEpollEvent {
                events: bits,
                u64: ev.udata as u64,
            });
        }

        Ok(rc as usize)
    }
}

impl AsRawFd for Epoll {
    fn as_raw_fd(&self) -> RawFd {
        self.kq
    }
}

impl Drop for Epoll {
    fn drop(&mut self) {
        // SAFETY: kq owned.
        unsafe {
            libc::close(self.kq);
        }
    }
}

fn empty_kevent() -> libc::kevent {
    libc::kevent {
        ident: 0,
        filter: 0,
        flags: 0,
        fflags: 0,
        data: 0,
        udata: std::ptr::null_mut(),
    }
}

fn make_kevent(fd: RawFd, filter: i16, flags: u16, udata: *mut c_void) -> libc::kevent {
    libc::kevent {
        ident: fd as usize,
        filter,
        flags,
        fflags: 0,
        data: 0,
        udata,
    }
}

// libc on macOS exposes EV_ERROR as i32 but `kevent.flags` is u16;
// keep a typed local constant so the casts are obvious.
const EV_ERROR: i32 = libc::EV_ERROR as i32;

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream;

    #[test]
    fn basic_register_and_wait() {
        let (a, b) = UnixStream::pair().unwrap();
        let epoll = Epoll::new().unwrap();
        epoll
            .ctl(
                ControlOperation::Add,
                b.as_raw_fd(),
                EpollEvent::new(EventSet::IN, 0xabcd),
            )
            .unwrap();

        // Nothing pending yet — short-timeout wait returns 0.
        let mut events = vec![EpollEvent::default(); 4];
        assert_eq!(epoll.wait(10, &mut events).unwrap(), 0);

        // Make `b` readable.
        use std::io::Write;
        (&a).write_all(&[42]).unwrap();

        let n = epoll.wait(100, &mut events).unwrap();
        assert_eq!(n, 1);
        assert!(events[0].event_set().contains(EventSet::IN));
        assert_eq!(events[0].data(), 0xabcd);
    }

    #[test]
    fn delete_unregisters() {
        let (_, b) = UnixStream::pair().unwrap();
        let epoll = Epoll::new().unwrap();
        epoll
            .ctl(
                ControlOperation::Add,
                b.as_raw_fd(),
                EpollEvent::new(EventSet::IN, 0),
            )
            .unwrap();
        epoll
            .ctl(
                ControlOperation::Delete,
                b.as_raw_fd(),
                EpollEvent::new(EventSet::IN, 0),
            )
            .unwrap();
    }
}
