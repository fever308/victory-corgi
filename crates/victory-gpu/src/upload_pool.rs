//! Bounded, size-bucketed pool of mapped D3D12 `UPLOAD` buffers for transient
//! texture staging (`crate::texture::Staging`).
//!
//! Eliminates steady-state `CreateCommittedResource` and `DxgkDestroyAllocation2`
//! churn (and NVIDIA driver lock contention between the submitting thread and
//! `victory-late-wait`) when streaming textures are staged, uploaded, and
//! unstaged/evicted.
//!
//! Safety invariant: returned upload buffers go through
//! [`crate::retire::retire_executing`] before re-entering the pool so a buffer
//! can never be handed to CPU detiling while any executing GPU submission could
//! still read from it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;

/// Default total pooled upload-buffer budget in MiB (`VICTORY_UPLOAD_POOL_MB`).
const DEFAULT_POOL_MB: u64 = 192;

/// Buffers larger than this are one-offs and bypass the pool.
const MAX_POOLED_BYTES: u64 = 32 * MIB;

/// Half-step (1x / 1.5x) power-of-two size classes from 64 KiB (the minimum
/// D3D12 committed buffer alignment) through 32 MiB. Worst-case capacity waste
/// above 64 KiB is <= 1.5x; any reuse of the next class is also capped at <= 2x
/// the requested size.
const SIZE_CLASSES: [u64; 19] = [
    64 * KIB,
    96 * KIB,
    128 * KIB,
    192 * KIB,
    256 * KIB,
    384 * KIB,
    512 * KIB,
    768 * KIB,
    MIB,
    1536 * KIB,
    2 * MIB,
    3 * MIB,
    4 * MIB,
    6 * MIB,
    8 * MIB,
    12 * MIB,
    16 * MIB,
    24 * MIB,
    32 * MIB,
];

fn pool_budget_bytes() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let mb = std::env::var("VICTORY_UPLOAD_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_POOL_MB);
        mb.saturating_mul(MIB)
    })
}

/// Smallest size class index whose byte capacity is `>= needed`, or `None` if
/// `needed > MAX_POOLED_BYTES`.
fn size_class_index(needed: u64) -> Option<usize> {
    let want = needed.max(1);
    if want > MAX_POOLED_BYTES {
        return None;
    }
    SIZE_CLASSES.iter().position(|&cap| cap >= want)
}

struct UploadEntry {
    buffer: ID3D12Resource,
    data: *mut u8,
    capacity: u64,
}

// SAFETY: D3D12 upload resources are free-threaded; the persistent CPU mapping
// is owned exclusively by the single `PooledUpload` or idle pool slot holding
// this entry, and only returned to the pool after executing GPU flights end.
unsafe impl Send for UploadEntry {}

struct IdleEntry {
    entry: UploadEntry,
    stamp: u64,
    returned_ms: u64,
}

struct PoolInner {
    budget: u64,
    total_bytes: u64,
    next_stamp: u64,
    buckets: [VecDeque<IdleEntry>; SIZE_CLASSES.len()],
}

impl PoolInner {
    fn new(budget: u64) -> Self {
        Self {
            budget,
            total_bytes: 0,
            next_stamp: 1,
            buckets: std::array::from_fn(|_| VecDeque::new()),
        }
    }

    /// Pops a cached buffer of size `>= needed` (exact class, or next class if
    /// within `2x` of `needed` so small textures never claim oversized buffers).
    fn pop(&mut self, needed: u64) -> Option<UploadEntry> {
        let idx = size_class_index(needed)?;
        if let Some(idle) = self.buckets[idx].pop_back() {
            self.total_bytes = self.total_bytes.saturating_sub(idle.entry.capacity);
            return Some(idle.entry);
        }
        let next = idx + 1;
        if next < SIZE_CLASSES.len() && SIZE_CLASSES[next] <= needed.saturating_mul(2)
            && let Some(idle) = self.buckets[next].pop_back()
        {
            self.total_bytes = self.total_bytes.saturating_sub(idle.entry.capacity);
            return Some(idle.entry);
        }
        None
    }

    /// Inserts `entry` into its bucket and evicts the oldest idle buffers across
    /// all buckets until `total_bytes <= budget`. Evicted resources are returned
    /// so the caller drops them outside the pool mutex.
    fn push(&mut self, entry: UploadEntry, now_ms: u64) -> Vec<UploadEntry> {
        let mut evicted = Vec::new();
        if self.budget == 0 || entry.capacity > self.budget {
            evicted.push(entry);
            return evicted;
        }
        let Some(idx) = size_class_index(entry.capacity) else {
            evicted.push(entry);
            return evicted;
        };
        let stamp = self.next_stamp;
        self.next_stamp = self.next_stamp.wrapping_add(1);
        self.total_bytes = self.total_bytes.saturating_add(entry.capacity);
        self.buckets[idx].push_back(IdleEntry {
            entry,
            stamp,
            returned_ms: now_ms,
        });
        while self.total_bytes > self.budget {
            let oldest_bucket = self
                .buckets
                .iter()
                .enumerate()
                .filter_map(|(i, q)| q.front().map(|front| (front.stamp, i)))
                .min_by_key(|&(stamp, _)| stamp)
                .map(|(_, i)| i);
            let Some(b) = oldest_bucket else {
                break;
            };
            if let Some(old) = self.buckets[b].pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(old.entry.capacity);
                evicted.push(old.entry);
            }
        }
        evicted
    }

    /// Evicts idle entries older than `max_age_ms` relative to `now_ms`.
    fn trim_older_than(&mut self, now_ms: u64, max_age_ms: u64) -> Vec<UploadEntry> {
        let mut evicted = Vec::new();
        for bucket in &mut self.buckets {
            while let Some(front) = bucket.front() {
                if now_ms.saturating_sub(front.returned_ms) >= max_age_ms {
                    if let Some(old) = bucket.pop_front() {
                        self.total_bytes = self.total_bytes.saturating_sub(old.entry.capacity);
                        evicted.push(old.entry);
                    }
                } else {
                    break;
                }
            }
        }
        evicted
    }
}

struct UploadPool {
    inner: Mutex<PoolInner>,
    hits: AtomicU64,
    misses: AtomicU64,
    returns: AtomicU64,
    evictions: AtomicU64,
}

impl UploadPool {
    fn new(budget: u64) -> Self {
        Self {
            inner: Mutex::new(PoolInner::new(budget)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            returns: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    fn acquire(self: &Arc<Self>, host: &crate::host::Host, needed: u64) -> Option<PooledUpload> {
        let want = needed.max(1);
        let budget = self.inner.lock().unwrap_or_else(|e| e.into_inner()).budget;
        if budget == 0 || want > MAX_POOLED_BYTES || want > budget {
            let (buffer, data) = host.upload_buffer(want)?;
            return Some(PooledUpload {
                entry: Some(UploadEntry {
                    buffer,
                    data,
                    capacity: want,
                }),
                pool: None,
            });
        }
        let cached = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.pop(want)
        };
        if let Some(entry) = cached {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(PooledUpload {
                entry: Some(entry),
                pool: Some(self.clone()),
            });
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let alloc_size = size_class_index(want).map_or(want, |i| SIZE_CLASSES[i]);
        let (buffer, data) = host.upload_buffer(alloc_size)?;
        Some(PooledUpload {
            entry: Some(UploadEntry {
                buffer,
                data,
                capacity: alloc_size,
            }),
            pool: Some(self.clone()),
        })
    }

    fn return_now(&self, entry: UploadEntry) {
        let evicted = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.push(entry, crate::host::now_ms())
        };
        self.returns.fetch_add(1, Ordering::Relaxed);
        if !evicted.is_empty() {
            self.evictions
                .fetch_add(evicted.len() as u64, Ordering::Relaxed);
        }
        // Drop evicted `ID3D12Resource`s outside the pool mutex so
        // `DxgkDestroyAllocation2` never stalls concurrent acquires/returns.
        drop(evicted);
    }

    fn trim_idle(&self, max_age_ms: u64) {
        let evicted = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.total_bytes == 0 {
                return;
            }
            inner.trim_older_than(crate::host::now_ms(), max_age_ms)
        };
        if !evicted.is_empty() {
            self.evictions
                .fetch_add(evicted.len() as u64, Ordering::Relaxed);
        }
        drop(evicted);
    }
}

fn global_pool() -> &'static Arc<UploadPool> {
    static POOL: OnceLock<Arc<UploadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let budget = pool_budget_bytes();
        tracing::info!(
            budget_mib = budget / MIB,
            max_buffer_mib = MAX_POOLED_BYTES / MIB,
            "texture upload buffer pool (VICTORY_UPLOAD_POOL_MB=0 disables)"
        );
        Arc::new(UploadPool::new(budget))
    })
}

/// Guard retired through [`crate::retire::retire_executing`] so a dropped
/// upload buffer only returns to its pool once all currently executing GPU
/// submissions have finished.
struct UploadReturnGuard {
    entry: Option<UploadEntry>,
    pool: Arc<UploadPool>,
}

impl Drop for UploadReturnGuard {
    fn drop(&mut self) {
        if let Some(entry) = self.entry.take() {
            self.pool.return_now(entry);
        }
    }
}

/// Mapped D3D12 `UPLOAD` buffer leased from the upload pool (or allocated
/// directly when pooling is disabled or the request exceeds `MAX_POOLED_BYTES`).
pub(crate) struct PooledUpload {
    entry: Option<UploadEntry>,
    pool: Option<Arc<UploadPool>>,
}

// SAFETY: `UploadEntry` is `Send`; `PooledUpload` exposes only `&ID3D12Resource`
// and raw `*mut u8` whose synchronization is governed by `Staging`.
unsafe impl Send for PooledUpload {}
// SAFETY: as above (`Staging` itself synchronizes writes to `data`).
unsafe impl Sync for PooledUpload {}

impl PooledUpload {
    pub(crate) fn buffer(&self) -> &ID3D12Resource {
        &self.entry.as_ref().expect("live pooled upload").buffer
    }

    pub(crate) fn data(&self) -> *mut u8 {
        self.entry.as_ref().expect("live pooled upload").data
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> u64 {
        self.entry.as_ref().expect("live pooled upload").capacity
    }
}

impl Drop for PooledUpload {
    fn drop(&mut self) {
        let Some(entry) = self.entry.take() else {
            return;
        };
        let Some(pool) = self.pool.take() else {
            drop(entry);
            return;
        };
        crate::retire::retire_executing(UploadReturnGuard {
            entry: Some(entry),
            pool,
        });
    }
}

/// Acquires a mapped D3D12 `UPLOAD` buffer of at least `needed` bytes from the
/// global pool, allocating one on miss.
pub(crate) fn acquire(host: &crate::host::Host, needed: u64) -> Option<PooledUpload> {
    global_pool().acquire(host, needed)
}

/// Trims idle pooled upload buffers that have sat unused for longer than the
/// idle window (`30 s` normally, `5 s` under RAM pressure).
pub(crate) fn trim_idle(ram_idle_ms: u64) {
    let max_age_ms = if ram_idle_ms < 5_000 { 5_000 } else { 30_000 };
    global_pool().trim_idle(max_age_ms);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Interface;

    #[test]
    fn size_class_rounding_and_bounds() {
        assert_eq!(size_class_index(0), Some(0));
        assert_eq!(size_class_index(1), Some(0));
        assert_eq!(SIZE_CLASSES[size_class_index(4096).unwrap()], 64 * KIB);
        assert_eq!(SIZE_CLASSES[size_class_index(64 * KIB).unwrap()], 64 * KIB);
        assert_eq!(
            SIZE_CLASSES[size_class_index(64 * KIB + 1).unwrap()],
            96 * KIB
        );
        assert_eq!(SIZE_CLASSES[size_class_index(100 * KIB).unwrap()], 128 * KIB);
        assert_eq!(
            SIZE_CLASSES[size_class_index(32 * MIB).unwrap()],
            32 * MIB
        );
        assert_eq!(size_class_index(32 * MIB + 1), None);
    }

    #[test]
    fn pool_reuses_same_size_class_and_respects_2x_waste_limit() {
        let _scope = crate::retire::isolated_retire_scope();
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = Arc::new(UploadPool::new(4 * MIB));

        // Allocate a 100 KiB upload -> rounds to 128 KiB class.
        let u1 = pool.acquire(host, 100 * KIB).expect("u1");
        let raw1 = u1.buffer().as_raw();
        assert_eq!(u1.capacity(), 128 * KIB);
        drop(u1);
        assert_eq!(pool.returns.load(Ordering::Relaxed), 1);

        // Requesting 1 KiB must NOT steal the 128 KiB buffer (128 KiB > 2 * 1 KiB,
        // and different class 64 KiB -> next class is 96 KiB > 2 * 1 KiB).
        let u_small = pool.acquire(host, 1024).expect("u_small");
        assert_ne!(u_small.buffer().as_raw(), raw1);
        assert_eq!(u_small.capacity(), 64 * KIB);

        // Requesting 90 KiB (96 KiB class) CAN reuse the 128 KiB buffer from the
        // adjacent class because 128 KiB <= 2 * 90 KiB.
        let u2 = pool.acquire(host, 90 * KIB).expect("u2");
        assert_eq!(u2.buffer().as_raw(), raw1);
        assert_eq!(pool.hits.load(Ordering::Relaxed), 1);

        drop(u2);
        drop(u_small);
    }

    #[test]
    fn pool_evicts_oldest_when_over_budget_and_bypasses_when_disabled() {
        let _scope = crate::retire::isolated_retire_scope();
        let Some(host) = crate::host::host() else {
            return;
        };
        // Budget = 256 KiB: holds at most two 128 KiB buffers.
        let pool = Arc::new(UploadPool::new(256 * KIB));

        let a = pool.acquire(host, 128 * KIB).expect("a");
        let b = pool.acquire(host, 128 * KIB).expect("b");
        let c = pool.acquire(host, 128 * KIB).expect("c");
        let (raw_a, raw_b, raw_c) = (
            a.buffer().as_raw(),
            b.buffer().as_raw(),
            c.buffer().as_raw(),
        );
        drop(a); // oldest (stamp 1)
        drop(b); // stamp 2
        drop(c); // stamp 3 -> must evict `a`!

        assert_eq!(pool.evictions.load(Ordering::Relaxed), 1);
        assert_eq!(pool.inner.lock().unwrap().total_bytes, 256 * KIB);

        // Popping LIFO yields `c`, then `b` (never evicted `a`).
        let p1 = pool.acquire(host, 128 * KIB).expect("p1");
        let p2 = pool.acquire(host, 128 * KIB).expect("p2");
        assert_eq!(p1.buffer().as_raw(), raw_c);
        assert_eq!(p2.buffer().as_raw(), raw_b);
        assert_ne!(p1.buffer().as_raw(), raw_a);
        assert_ne!(p2.buffer().as_raw(), raw_a);
        drop((p1, p2));

        // Disabled pool (`budget == 0`) allocates exact size and never caches.
        let disabled = Arc::new(UploadPool::new(0));
        let d = disabled.acquire(host, 4096).expect("disabled");
        assert_eq!(d.capacity(), 4096);
        assert!(d.pool.is_none());
        drop(d);
        assert_eq!(disabled.returns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pooled_upload_waits_for_active_exec_epoch_before_returning_to_pool() {
        let _scope = crate::retire::isolated_retire_scope();
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = Arc::new(UploadPool::new(2 * MIB));

        let u1 = pool.acquire(host, 128 * KIB).expect("u1");
        let raw1 = u1.buffer().as_raw();

        // While a GPU execution epoch is active, dropping `u1` must NOT return
        // its buffer to the pool yet (GPU may still be reading from it).
        let exec = crate::retire::acquire_exec_guard();
        drop(u1);
        assert_eq!(
            pool.returns.load(Ordering::Relaxed),
            0,
            "buffer must not return to pool while GPU execution epoch is active"
        );
        assert_eq!(pool.inner.lock().unwrap().total_bytes, 0);

        // Acquiring now must allocate a distinct buffer, not `raw1`.
        let u2 = pool.acquire(host, 128 * KIB).expect("u2");
        assert_ne!(u2.buffer().as_raw(), raw1);

        // Once the execution epoch finishes, `raw1` immediately returns to the pool.
        drop(exec);
        assert_eq!(pool.returns.load(Ordering::Relaxed), 1);
        assert_eq!(pool.inner.lock().unwrap().total_bytes, 128 * KIB);

        // Next acquire reuses `raw1`.
        let u3 = pool.acquire(host, 128 * KIB).expect("u3");
        assert_eq!(u3.buffer().as_raw(), raw1);
        drop((u2, u3));
    }
}
