//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). the title's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** use a scalar integer UAV. The host binds R32_UINT
//!   for 32-bit operations and R32G32_UINT for 64-bit operations. The latter
//!   require SM 6.6 atomic support on typed and descriptor-heap resources.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).
//! * **Rectangle lists** ([`rect_list_gs`], `gs_6_6`): the console draws a
//!   rectangle from three vertices; the host has no such topology. A
//!   pipeline the title makes for rectangles (topology type 19) gets a
//!   geometry shader of ours between the translated vertex and pixel
//!   shaders and draws triangle lists with the title's own vertex counts.
//!   It binds nothing.

mod bitcode;
mod container;
mod lds_order;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod intersect_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod own_bits;
mod rect;
mod sampler;
mod shader;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
#[cfg(all(test, windows))]
mod wave64_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use bitcode::census;
pub use sampler::StaticSampler;
pub use lower::{set_host_max_wave, with_trace};
pub use rect::rect_list_gs;
pub use shader::{
    MEMORY_FAULTS, MEMORY_PAGES, ROOT_CBV_SPACE, SAMPLER_SPACE, SLOT_FAULTS, SLOT_MEMORY,
    SLOT_STRIDES, SLOT_STRIDES_COUNT, SLOT_TRACE, SLOT_VIEW_MARKS, SLOT_VIEW_WRITES, TRACE_FIRST,
    TRACE_PIXEL, TRACE_THREAD, TRACE_WORDS, USER_DATA_SPACE, USER_GROUP_BASE, USER_MEMORY_BASE,
    USER_MESH_ROW, USER_ROOT_LIMIT, USER_STRUCTURES, USER_TRACE, USER_VIEW_ADDRESS, USER_VIEW_BIAS,
    VIEW_MARK_NONE, VIEW_WRITE_BITS, set_view_heap_limit, view_heap_limit,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// Which of a translated program's group barriers wait for the whole
/// group (`GroupMemoryBarrierWithGroupSync`); the others keep their memory
/// fence and lose the wait. A wait that some thread of the group never
/// reaches is undefined on the host, and NVIDIA's wait is for every
/// thread: the group stands until Windows resets the card. (The console's
/// barrier is the wave's, whatever lanes are live, and a wave that ended
/// is not waited for.) `VICTORY_GROUP_SYNC`:
/// - unset or `2`: every barrier waits, as before.
/// - `1`: the title's `s_barrier` waits only in a block every lane runs,
///   in a program no lane can be cut out of at the iteration limit.
/// - `0`: no barrier waits, neither the title's nor the ones the
///   translation adds (wave64 programs in two host waves exchange through
///   them, so such programs compute wrong values). For one run that tells
///   whether a barrier is what stops the card: with it no run can stand in
///   a barrier.
///
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn group_sync() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GROUP_SYNC").ok().as_deref().map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        _ => 2,
    })
}

/// The clean-up passes that run over a translated program before it is
/// written (`bitcode::opt`): a set of that module's pass bits. Every pass
/// is exact, so the switch is for telling a translator fault from a
/// clean-up fault, and for measuring what the clean-up is worth.
/// `VICTORY_SHADER_OPT`:
/// - unset: every pass.
/// - `0`: none (the module as the lowering made it).
/// - a number (decimal or `0x` hex): those passes; 1 dead code, 2 constants
///   and identities, 4 one instruction for one value, 8 what a branch taken
///   says about its condition, 16 one read for one place within a block,
///   32 a first-lane read of two host waves without the round that asks
///   which half has a lane (where every lane has one).
///
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn shader_opt() -> u32 {
    static PASSES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| parse_shader_opt(std::env::var("VICTORY_SHADER_OPT").ok().as_deref()))
}

/// Parses `VICTORY_INDIRECT_AMPLIFY`: enabled by default; explicit `"0"`,
/// `"false"`, `"no"`, or `"off"` (ASCII case-insensitive) disables the
/// amplification stage for one-record indirect mesh draws.
pub fn parse_indirect_amplify(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(value) = value else {
        return true;
    };
    let Some(text) = value.to_str() else {
        return true;
    };
    let trimmed = text.trim();
    !(trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off"))
}

/// Whether one-record indirect mesh draws use our amplification stage
/// (`VICTORY_INDIRECT_AMPLIFY=0` disables).
///
/// Part of the translation: the shader cache keys on it (`translation_fingerprint`).
pub fn indirect_amplify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_indirect_amplify(std::env::var_os("VICTORY_INDIRECT_AMPLIFY").as_deref())
    })
}

/// The payload bytes mesh shaders declare: 4 when
/// [`indirect_amplify_enabled`] (`VICTORY_INDIRECT_AMPLIFY=0` disables;
/// `victory-gpu` puts an amplification stage of its own before them, whose
/// payload is one word they do not read; the host wants the two sizes to
/// agree), else 0.
pub(crate) fn mesh_payload() -> u32 {
    if indirect_amplify_enabled() { 4 } else { 0 }
}

/// Whether translated shaders emit 64 KiB page write marks into the host's
/// view-write bitmap (`SLOT_VIEW_MARKS` / `SLOT_VIEW_WRITES`).
/// `VICTORY_VIEW_MARKS=1` enables it; unset or `0` leaves translated shaders
/// and copy-back behavior completely unchanged.
///
/// Part of the translation: the shader cache keys on it (`translation_fingerprint`).
pub fn view_marks_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_MARKS").is_some_and(|v| v == "1"))
}

/// Everything a translation depends on besides the container: the
/// translator's own source (hashed at build time), the value of every
/// `VICTORY_*` switch it reads, the host facts the device hands it (widest
/// wave, view heap limit) and the settings above. The shader cache keys on
/// this instead of on the whole executable, so a rebuild that leaves the
/// translator alone keeps its translations. Call it after the device is made.
pub fn translation_fingerprint() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(env!("VICTORY_SHADER_SOURCE_HASH").as_bytes());
    out.push(u8::from(cfg!(feature = "diagnostics")));
    out.push(u8::from(cfg!(debug_assertions)));
    for name in env!("VICTORY_SHADER_SWITCHES").split(',').filter(|n| !n.is_empty()) {
        out.extend_from_slice(name.as_bytes());
        out.push(b'=');
        if let Some(value) = std::env::var_os(name) {
            out.extend_from_slice(value.as_encoded_bytes());
        }
        out.push(0);
    }
    out.extend_from_slice(&lower::host_max_wave().to_le_bytes());
    out.extend_from_slice(&shader::view_heap_limit().to_le_bytes());
    out.extend_from_slice(&loop_limit().to_le_bytes());
    out.push(group_sync());
    out.extend_from_slice(&shader_opt().to_le_bytes());
    out.push(u8::from(view_marks_enabled()));
    out.push(u8::from(indirect_amplify_enabled()));
    out
}

/// `VICTORY_SHADER_OPT`'s value; every pass when it is unset or
/// unreadable. Pure, for testing.
fn parse_shader_opt(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    parsed.map_or(bitcode::opt::ALL, |n| n & bitcode::opt::ALL)
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
    /// Heap descriptors that can be selected from root state at bind time.
    pub root_indexed_views: Vec<crate::resource::RootIndexedView>,
    /// Compute: host thread groups per console thread group, along z (1: the console's
    /// groups as they are). With n > 1 each console group of n 64-lane waves runs as n host
    /// groups of one wave: the dispatch's z count is multiplied by n.
    pub group_split: u32,
    /// Table descriptors (root parameter, dword offset in the table) whose
    /// buffer the shader reads or writes sixteen bits a channel
    /// (`buffer_load_format_d16_*`, `buffer_store_format_d16_*`), in
    /// order. On the console each channel's 16 bits move as they are
    /// through a 16-bit float view and through a 16-bit integer one alike
    /// (the title keeps half floats, and indices beside them, in
    /// `R16G16B16A16_UINT` and `R16_UINT` buffers). The DXIL moves them as
    /// unsigned integers, which an unsigned 16-bit view carries unchanged;
    /// the host shows the shader a 16-bit float or signed buffer view at
    /// these places as the unsigned format of the same layout.
    /// shortcut: a view in a normalised 16-bit format, whose values the
    /// console would convert to halves, is read as its integers; a
    /// descriptor picked at run time keeps the float path (its slot, and so
    /// its format, is known only then).
    pub half_views: Vec<(u8, u32)>,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }

    /// Whether this shader marks the 64 KiB pages it writes through unordered
    /// buffer views (`SLOT_VIEW_MARKS` / `SLOT_VIEW_WRITES`).
    pub fn has_view_marks(&self) -> bool {
        self.root_uavs.contains(&SLOT_VIEW_WRITES)
    }
}

/// How a translation is made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Writes through unordered buffer views mark their 64 KiB pages in
    /// `u<SLOT_VIEW_WRITES>` using `t<SLOT_VIEW_MARKS>`.
    pub view_marks: bool,
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
///
/// A program refused because the first half of an interpolation is read as
/// data is translated once more with first halves evaluated as the pair
/// they start (`lower::with_first_half_pairs`), and only such a program:
/// with that on for every program, 8,551 pixel programs of the title that
/// already translated came out changed (an extra evaluation in each), for
/// the sake of 9 that did not translate.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_for_hash(container, "")
}

/// As [`translate`], with the 16-hex-character container hash available for
/// shader-specific Wave32 exact-lowering overrides (such as `AlphaCBPPPC`).
pub fn translate_for_hash(container: &[u8], hash: &str) -> Result<HostShader, ShaderError> {
    translate_with_hash(
        container,
        Options {
            view_marks: view_marks_enabled(),
        },
        hash,
    )
}

/// [`translate`] made as `options` say.
pub fn translate_with(container: &[u8], options: Options) -> Result<HostShader, ShaderError> {
    translate_with_hash(container, options, "")
}

/// [`translate_with`] with the container hash available.
pub fn translate_with_hash(
    container: &[u8],
    options: Options,
    hash: &str,
) -> Result<HostShader, ShaderError> {
    match translate_plain(container, options, hash) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::FIRST_HALF => {
            lower::with_first_half_pairs(|| translate_plain(container, options, hash))
        }
        other => other,
    }
}

fn translate_plain(container: &[u8], options: Options, hash: &str) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let p = crate::spill::build(&native)?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit_with_hash(&c, &native, &p, &st, &io, &res, options, hash)
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with_hash(
        c,
        native,
        p,
        st,
        io,
        res,
        Options {
            view_marks: view_marks_enabled(),
        },
        "",
    )
}

/// [`emit`] made as `options` say.
pub fn emit_with(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: Options,
) -> Result<HostShader, ShaderError> {
    emit_with_hash(c, native, p, st, io, res, options, "")
}

thread_local! {
    /// Wave-wide branches for the program being translated on this thread
    /// ([`with_wave_wide`]), with `VICTORY_WAVE_WIDE` unset.
    static WIDE_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn parse_wave_wide_only(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(',')
        .map(|h| h.trim().to_lowercase())
        .filter(|h| !h.is_empty())
        .collect()
}

fn matches_wave_wide_only(hash: &str, only: &[String]) -> bool {
    if only.is_empty() {
        return false;
    }
    let hash = hash.to_lowercase();
    only.iter().any(|o| hash.starts_with(o.as_str()))
}

fn wave_wide_only_list() -> &'static [String] {
    static ONLY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ONLY.get_or_init(|| {
        parse_wave_wide_only(std::env::var("VICTORY_WAVE_WIDE_ONLY").ok().as_deref())
    })
}

/// Whether `VICTORY_WAVE_WIDE_ONLY=<hash prefix>,...` names the container whose hash (as a
/// run's log names it: the first 16 hex digits of its BLAKE3) is `hash`. Those programs alone
/// get the wave-wide form that `VICTORY_WAVE_WIDE=1` gives every program: a console wave64
/// program in thread groups of one console wave, whose per-lane branches around its
/// cross-lane code are taken by the whole wave ([`widen`]). The form is off by default
/// outside verified hashes because it once left a dispatch on the card until the device was
/// reset, so it is turned on only for programs checked one by one.
pub fn wave_wide_only(hash: &str) -> bool {
    matches_wave_wide_only(hash, wave_wide_only_list())
}

/// Runs `f` (a translation on this thread) with the wave-wide form on when `on`
/// ([`wave_wide_only`]).
pub fn with_wave_wide<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let before = WIDE_HERE.replace(on);
    let r = f();
    WIDE_HERE.set(before);
    r
}

fn wide_lowering_enabled(
    hash: &str,
    wide_env: Option<&std::ffi::OsStr>,
    wide_only: &[String],
    wide_here: bool,
    group_sync: u8,
) -> bool {
    if wide_env.is_some_and(|v| v == "1") {
        return true;
    }
    if wide_env.is_some_and(|v| v == "0") {
        return false;
    }
    if !wide_only.is_empty() {
        return wide_here || matches_wave_wide_only(hash, wide_only);
    }
    if wide_here {
        return true;
    }
    // Verified water-shading compute hashes safe with wave-wide lowering
    // (includes a10179ab7eda0dc3 = CS_TransmissionVolumetrics_VolumePortalsOnly
    // / Submerged from fxceless/victory-fork#1248).
    // Note: 5dc3b25986622949 and 6f225b641c5b0708 (_Tiled / _Tiled_LocalLights)
    // are intentionally excluded (caused black rectangles around the boat).
    matches!(
        hash,
        "6871d6e4b3ac9a10"
            | "926c46bab3b86bd6"
            | "a10179ab7eda0dc3"
            | "369cbe906ea72764"
            | "2cc4c197ee29e291"
    ) && group_sync != 0
}

/// [`emit_with`] with the container hash available.
#[allow(clippy::too_many_arguments)]
pub fn emit_with_hash(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: Options,
    hash: &str,
) -> Result<HostShader, ShaderError> {
    // (what an earlier program's lowering on this thread left behind)
    lower::take_needs();
    let original = emit_program(c, native, p, st, io, res, options);
    // A console wave64 compute program on a host of 32-lane waves that
    // still needs 64 lanes, or whose cross-lane code sits where not every
    // lane runs: once more with its early-outs left out
    // ([`ir::without_early_outs`]), which is what usually keeps the whole
    // body from being a block every lane runs.
    let stuck = match &original {
        Ok((_, wave)) => *wave == Some(64),
        Err(ShaderError::Unsupported { what, .. }) => lower::needs_every_lane(what),
        Err(_) => false,
    };
    // (a mask with lanes elsewhere is only helped by the exact forms: not by each half-wave
    // on its own)
    let exact_only = matches!(&original, Err(ShaderError::Unsupported { what, .. }) if *what == lower::MASK_ELSEWHERE);
    if stuck && native.stage == HwStage::Cs && native.wave64 && lower::host_max_wave() == 32 {
        let why = std::env::var_os("VICTORY_WAVE_WHY").is_some();
        let report = |what: &str, r: &Result<(HostShader, Option<u32>), ShaderError>| {
            if why {
                match r {
                    Ok((_, wave)) => eprintln!("WAVE64-RETRY {what}: wave {wave:?}"),
                    Err(e) => eprintln!("WAVE64-RETRY {what}: {e}"),
                }
            }
        };
        let done = |r: &Result<(HostShader, Option<u32>), ShaderError>| matches!(r, Ok((_, wave)) if *wave != Some(64));
        // Each host wave as a wave of its own (`lower::with_local_waves`),
        // for programs still stopped where not every lane runs.
        let local_on = std::env::var_os("VICTORY_WAVE_LOCAL").is_none_or(|v| v != "0");
        // Where the wave operations are the allocation idiom
        // ([`lower::wave_local`]) that is exact. Other programs are taken
        // the same way last of all, each half-wave giving its own results
        // (approximate: a tile-wide result becomes two half-tile ones);
        // `VICTORY_WAVE_LOCAL=idiom` refuses those instead.
        let strict = std::env::var_os("VICTORY_WAVE_LOCAL").is_some_and(|v| v == "idiom");
        let idiom = |p: &ir::Program, st: &structure::Structured| match lower::wave_local(p, st) {
            Ok(()) => true,
            Err((offset, what)) => {
                if why {
                    eprintln!("WAVE64-RETRY local waves: not the allocation idiom at {offset:#x}: {what}");
                }
                false
            }
        };
        // The program in two other forms: without its early-outs, and
        // without any branch that only skips code no lane is left for.
        let form = |p: Option<ir::Program>| {
            p.and_then(|p| {
                let st = structure::structure(&p).ok()?;
                let io = io::raise(&p, native).ok()?;
                let res = resource::track(&p, &native.layout, native.stage, Some(&st)).ok()?;
                Some((p, st, io, res))
            })
        };
        let forms = [
            ("without early-outs", form(ir::without_early_outs(p))),
            ("without skips", form(ir::without_skips(p, false))),
        ];
        for (name, f) in &forms {
            if let Some((p, st, io, res)) = f {
                let again = emit_program(c, native, p, st, io, res, options);
                report(name, &again);
                if done(&again) {
                    return again.map(|(shader, _)| shader);
                }
            }
        }
        // For a console wave run as two host waves this form puts group barriers inside
        // branches and loops. Keep it opt-in outside the checked shaders: runs 121-124
        // lost the card in a submission of two programs that had just moved to it
        // (`ptfx_CS_lightAtlas_Sprite` and `_Trail`). The 32-lane programs' form below, which
        // has no barrier, stays on.
        let wide = std::env::var_os("VICTORY_WAVE_WIDE");
        let wide_on = wide_lowering_enabled(
            hash,
            wide.as_deref(),
            wave_wide_only_list(),
            WIDE_HERE.get(),
            group_sync(),
        );
        // exact first: each host wave on its own where that is exact (the idiom, every
        // form; no barriers), then the other exact forms; then approximate
        for exact in [true, false] {
            if !exact {
                // Before anything approximate, the exact forms that cost group barriers.
                // Every skip left out that the code after it cannot see
                // ([`ir::without_skips`], `deep`): most of the body is then code every
                // lane runs, where the two host waves can exchange.
                // (`VICTORY_WAVE_DEEP_SKIPS=0` and `VICTORY_WAVE_WIDE=0` turn the two off.)
                let deep_on = std::env::var_os("VICTORY_WAVE_DEEP_SKIPS").is_none_or(|v| v != "0");
                let deep = if deep_on { form(ir::without_skips(p, true)) } else { None };
                if let Some((p, st, io, res)) = &deep {
                    let again = emit_program(c, native, p, st, io, res, options);
                    report("without skips, scalars followed", &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
                // Still stopped where not every lane runs: the per-lane branches around
                // those places taken as the console takes them, by the whole wave at once
                // ([`widen`]), at two group barriers per such branch.
                if wide_on {
                    let bases = [
                        deep.as_ref().map(|f| &f.0),
                        forms[1].1.as_ref().map(|f| &f.0),
                        forms[0].1.as_ref().map(|f| &f.0),
                        Some(p),
                    ];
                    if let Some(base) = bases.into_iter().flatten().next() {
                        let again = widen(c, native, base, why, options);
                        report("wave-wide branches", &again);
                        if done(&again) {
                            return again.map(|(shader, _)| shader);
                        }
                    }
                }
                // AlphaCBPPPC uses Wave64 across divergent branches, but its two
                // waves share no LDS or group barrier. Split them into separate
                // host groups (`group_split = 2`) before the approximate local-waves
                // fallback can take over on Wave32 hosts (after upstream fedad73b).
                if hash == "f133e35efed2a48a"
                    && group_sync() != 0
                    && std::env::var_os("VICTORY_CASCADE_SPLIT_WAVE").is_none_or(|v| v != "0")
                    && matches!(&native.io, crate::native::StageIo::Cs(cs)
                        if cs.threads == [8, 16, 1] && cs.lds_bytes() == 0)
                    && !p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_barrier")
                {
                    let again = widen(c, native, p, why, options);
                    report("one console wave per host group (AlphaCBPPPC)", &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
                if strict {
                    break;
                }
            }
            if !local_on || exact_only {
                continue;
            }
            for (name, f) in &forms {
                if let Some((p, st, io, res)) = f
                    && idiom(p, st) == exact
                {
                    let local = lower::with_local_waves(|| emit_program(c, native, p, st, io, res, options));
                    report(&format!("local waves ({}), {name}", if exact { "idiom" } else { "APPROXIMATE" }), &local);
                    if done(&local) {
                        return local.map(|(shader, _)| shader);
                    }
                }
            }
            if idiom(p, st) == exact {
                let local = lower::with_local_waves(|| emit_program(c, native, p, st, io, res, options));
                report(if exact { "local waves (idiom)" } else { "local waves (APPROXIMATE)" }, &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
        if local_on && !exact_only {
            // Last: every mask and lane read per host wave.
            if !strict {
                for (name, f) in &forms {
                    if let Some((p, st, io, res)) = f {
                        let local = lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res, options));
                        report(&format!("local waves (APPROXIMATE, any mask), {name}"), &local);
                        if done(&local) {
                            return local.map(|(shader, _)| shader);
                        }
                    }
                }
                let local = lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res, options));
                report("local waves (APPROXIMATE, any mask)", &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
        // Only when asked for (`VICTORY_WAVE_LOOSE_LDS=1`), since it can be wrong: LDS read
        // and written inside one lane-branch region without a barrier there even where
        // `lds_order` cannot prove that no thread sees another's write
        // ([`lower::with_loose_lds`]), first as the program is, then in its other forms,
        // then with each host wave on its own. (Run 20261124-174052 drew streaks across
        // the screen with it on, run -174823 none with it off.)
        if !strict && std::env::var_os("VICTORY_WAVE_LOOSE_LDS").is_some_and(|v| v == "1") {
            let again = lower::with_loose_lds(|| emit_program(c, native, p, st, io, res, options));
            report("LDS without barriers in lane branches (APPROXIMATE)", &again);
            if done(&again) {
                return again.map(|(shader, _)| shader);
            }
            for (name, f) in &forms {
                if let Some((p, st, io, res)) = f {
                    let again = lower::with_loose_lds(|| emit_program(c, native, p, st, io, res, options));
                    report(&format!("LDS without barriers in lane branches (APPROXIMATE), {name}"), &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
            }
            if local_on && !exact_only {
                let local = lower::with_loose_lds(|| {
                    lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res, options))
                });
                report("LDS without barriers in lane branches, local waves (APPROXIMATE, any mask)", &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
        // Last resort, for a program no other form translates (it was dropped): its per-lane
        // branches taken wave-wide ([`widen`]), which is exact. Off as the general order
        // above unless asked for (`VICTORY_WAVE_WIDE=1`), since moving programs that did
        // translate to it lost the card (runs 121-124); here only programs that would
        // otherwise not run at all take it. `VICTORY_WAVE_WIDE=0` turns this off too.
        // (the title's a53a7977fe762268, dispatched twice a frame: its light
        // loops, wave ORs over the lanes' cluster masks, sit in lane branches and needed the
        // two host waves to exchange there; 8 branches taken wave-wide.)
        let wide_last = !wide_on && wide.as_deref().is_none_or(|v| v != "0");
        if wide_last {
            let again = widen(c, native, p, why, options);
            report("wave-wide branches, last resort", &again);
            if done(&again) {
                return again.map(|(shader, _)| shader);
            }
            // the program's other forms, whose per-lane descriptors may track where the
            // program's own do not with its branches taken wave-wide
            for (name, f) in &forms {
                if let Some((p, ..)) = f {
                    let again = widen(c, native, p, why, options);
                    report(&format!("wave-wide branches, last resort, {name}"), &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
            }
        }
    }
    // SCC of a mask operation joined with a scalar test behind a per-lane skip (the 32-lane
    // trace programs): right once the wave takes that skip together ([`widen`]; the
    // lowering has named the skipped blocks).
    let original = match original {
        Err(ShaderError::Unsupported { what, .. })
            if native.stage == HwStage::Cs
                && what == lower::LANE_MASK
                && std::env::var_os("VICTORY_WAVE_WIDE").is_none_or(|v| v != "0") =>
        {
            let again = widen(c, native, p, std::env::var_os("VICTORY_WAVE_WHY").is_some(), options);
            match again {
                Ok(_) => again,
                Err(_) => emit_program(c, native, p, st, io, res, options),
            }
        }
        other => other,
    };
    let original = original.map(|(shader, _)| shader);
    // Retry unsupported programs only. Threading can separate an unknown
    // register value from its apparent use on a mutually exclusive path.
    if !matches!(original, Err(ShaderError::Unsupported { .. })) {
        return original;
    }
    let edges = threading::constant_edges(p, st);
    if edges.is_empty() {
        return original;
    }
    let p = ir::thread_edges(p, &edges)?;
    let st = structure::structure(&p)?;
    let io = io::raise(&p, native)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    emit_program(c, native, &p, &st, &io, &res, options).map(|(shader, _)| shader)
}

/// `p` (a console wave64 compute program on a host of 32-lane waves) with the per-lane
/// branches around its cross-lane code taken wave-wide ([`structure::structure_wide`]): the
/// lowering names the blocks where the two host waves must meet and could not
/// (`lower::take_needs`), the branches whose regions hold them are widened, and so on until
/// it translates or a branch cannot be widened. Only where a thread group is one console
/// wave: with several, the waves of a group would decide differently and a barrier inside
/// such a branch would wait for threads that never come.
fn widen(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    why: bool,
    options: Options,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    const NOT: ShaderError = ShaderError::Unsupported {
        offset: 0,
        what: "wave-wide branches in a thread group of several console waves",
    };
    let mut split = false;
    match native.io {
        // (a program whose waves are the host's meets no one: no barrier, any group)
        crate::native::StageIo::Cs(_) if !(native.wave64 && lower::host_max_wave() == 32) => {}
        crate::native::StageIo::Cs(cs) if cs.threads.iter().product::<u32>() == 64 => {}
        // Waves of one group that share nothing – no LDS, no barrier – never meet on the
        // console either: each runs as its own host group (`lower::with_group_split`).
        crate::native::StageIo::Cs(cs)
            if group_split_on()
                && cs.threads.iter().product::<u32>() % 64 == 0
                && cs.lds_bytes() == 0
                && !p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_barrier") =>
        {
            if why {
                eprintln!("WAVE64-WIDE thread groups of {:?} split into one host group per console wave", cs.threads);
            }
            split = true;
        }
        // Several console waves that share LDS or meet at barriers: the two host waves of a
        // console wave meet on their own inside the branches it takes together, not at group
        // barriers the group's other waves would have to come to (`lower::Lower::meet`).
        crate::native::StageIo::Cs(cs)
            if lower::pair_sync_on() && cs.threads.iter().product::<u32>() % 64 == 0 =>
        {
            if why {
                eprintln!("WAVE64-WIDE thread groups of {:?}: the halves of each console wave meet on their own", cs.threads);
            }
        }
        crate::native::StageIo::Cs(cs) => {
            if why {
                eprintln!("WAVE64-WIDE not tried: thread groups of {:?}", cs.threads);
            }
            return Err(NOT);
        }
        _ => return Err(NOT),
    }
    if split {
        return lower::with_group_split(|| widen_loop(c, native, p, why, options));
    }
    widen_loop(c, native, p, why, options)
}

/// `VICTORY_GROUP_SPLIT=0`: thread groups of several console waves are not split for
/// wave-wide branches ([`widen`]).
fn group_split_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GROUP_SPLIT").is_none_or(|v| v != "0"))
}

fn widen_loop(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    why: bool,
    options: Options,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let mut wide = std::collections::HashSet::new();
    // The lanes' own paths, as the console takes them: what descriptors the lanes reaching an
    // access hold follows from those, not from the branches the host takes wave-wide (a
    // descriptor loaded inside a skip and read after its join: only lanes that ran the skip
    // read it, which a wave-wide branch no longer says).
    let lane_paths = structure::structure(p).ok();
    loop {
        let st = structure::structure_wide(p, &wide)?;
        let io = io::raise(p, native)?;
        let res = match resource::track(p, &native.layout, native.stage, Some(&st)) {
            Ok(res) => res,
            Err(e) => match &lane_paths {
                Some(lanes) => resource::track(p, &native.layout, native.stage, Some(lanes)).map_err(|_| e)?,
                None => return Err(e),
            },
        };
        lower::take_needs();
        let r = emit_program(c, native, p, &st, &io, &res, options);
        let needs = lower::take_needs();
        // (or translated for 64-lane waves still: shared memory read and written where not
        // every lane runs, whose blocks the lowering names too)
        let stuck = matches!(&r, Err(ShaderError::Unsupported { what, .. })
                if lower::needs_every_lane(what) || *what == lower::LANE_MASK && !needs.is_empty())
            || matches!(&r, Ok((_, Some(64)))) && !needs.is_empty();
        if !stuck {
            if why {
                eprintln!("WAVE64-WIDE {} branches taken wave-wide", wide.len());
            }
            return r;
        }
        let more: Vec<usize> = lower::branches_around(&st, &needs)
            .unwrap_or_default()
            .into_iter()
            .filter(|b| !wide.contains(b))
            .collect();
        if more.is_empty() {
            if why {
                eprintln!("WAVE64-WIDE stopped with {} branches wave-wide: a branch around it cannot be", wide.len());
            }
            return r;
        }
        wide.extend(more);
    }
}

/// The translated program, and the wave size it requires of the host.
fn emit_program(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: Options,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (the title's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        view_marks: options.view_marks,
    })?;
    let wave_size = l.wave_size;
    let shader = HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
        group_split: l.group_split,
        half_views: l.half_views,
    };
    Ok((shader, wave_size))
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_clean_up_switch_parses_or_runs_every_pass() {
        let all = super::bitcode::opt::ALL;
        assert_eq!(parse_shader_opt(None), all);
        assert_eq!(parse_shader_opt(Some("0")), 0);
        assert_eq!(parse_shader_opt(Some(" 1 ")), 1);
        assert_eq!(parse_shader_opt(Some("0x1")), 1);
        assert_eq!(parse_shader_opt(Some("0xffffffff")), all);
        assert_eq!(parse_shader_opt(Some("on")), all);
    }

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    #[test]
    fn immediate_sampler_slots_distinguish_comparison_uses() {
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0000,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[0:1], null
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E00_02F0, // v_mov_b32 v0, 0.5 (comparison reference)
            0x7E02_02F0, // v_mov_b32 v1, 0.5 (u)
            0x7E04_02F0, // v_mov_b32 v2, 0.5 (v)
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xF080_0108,
            0x0040_0301, // image_sample v3, v[1:2], s[0:7], s[8:11]
            0xF0A0_0108,
            0x0040_0400, // image_sample_c v4, v[0:2], s[0:7], s[8:11]
            0xF0BC_0108,
            0x0040_0500, // image_sample_c_lz v5, v[0:2], s[0:7], s[8:11]
            0xF080_0108,
            0x0040_0601, // image_sample v6, v[1:2], s[0:7], s[8:11]
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_180F,
            0x0605_0403, // exp mrt0, v3, v4, v5, v6 done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 7,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let output = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let osg1 = container::signature(&[output], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();

        let sampler = StaticSampler::from_words([0; 4]);
        let comparison = StaticSampler {
            filter: sampler.filter | 0x80,
            ..sampler
        };
        assert_eq!(out.samplers, [[0; 4]; 2]);
        assert_eq!(out.static_samplers, [sampler, comparison]);
        #[cfg(windows)]
        if let Some(path) = super::validator::Validator::default_path() {
            let mut v = super::validator::Validator::load(Some(&path)).unwrap();
            v.validate(&mut out.dxil.clone()).unwrap();
        }
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask was
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now that low dword, seen with two high
    /// dwords, has no mask, and the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A wave64 compute program: `body`, then a store of v1 to a UAV.
    fn stores_v1(body: &[u32]) -> Result<(), ShaderError> {
        use crate::native::RootParam;
        use crate::{ResourceType, RootType, UsedResource};
        // s_mov_b32 s1, 0; s_load_dwordx4 s[8:11], s[0:1], 0
        let mut words = vec![0xBE81_0380, 0xF408_0200, 0xFA00_0000];
        words.extend_from_slice(body);
        // v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
        // s_endpgm
        words.extend_from_slice(&[0x3406_0082, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        emit(&c, &native, &p, &st, &io, &res).map(|_| ())
    }

    /// The high dword of a wave64 mask rewritten alone (`s_mov_b32 vcc_hi`
    /// or `exec_hi`): lanes 32-63 took the mask the low dword was first
    /// written with (the compare's, EXEC's on entry). Now the store refuses;
    /// without the write it translates.
    #[test]
    fn a_mask_whose_high_dword_is_rewritten_is_refused() {
        // v_cmp_eq_u32 vcc, 0, v0; v_mov_b32 v2, 11; v_cndmask_b32 v1, 0, v2, vcc
        let select = [0x7D84_0080, 0x7E04_028B, 0x0202_0480];
        // The same with v_add_co_ci_u32 v1, vcc, 0, v2, vcc.
        let carry = [0x7D84_0080, 0x7E04_028B, 0x5002_0480];
        // v_mov_b32 v1, 7
        let mov = [0x7E02_0287];
        for body in [&select[..], &carry, &mov] {
            stores_v1(body).unwrap();
        }
        // s_mov_b32 vcc_hi, 0 after the compare.
        let select_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x0202_0480];
        let carry_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x5002_0480];
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, 7, then the same after
        // s_mov_b64 s[2:3], exec.
        let exec_hi = [0xBEFF_0380, 0x7E02_0287];
        let exec_hi_saved = [0xBE82_047E, 0xBEFF_0380, 0x7E02_0287];
        for body in [&select_hi[..], &carry_hi, &exec_hi, &exec_hi_saved] {
            assert!(stores_v1(body).is_err(), "{body:x?}");
        }
    }

    /// `ds_bpermute_b32` stores nothing: each lane reads the lane its address
    /// names. It stopped as "LDS the program does not allocate" (sticky lips)
    /// or "an LDS operation" (the checkerboard shadow masks).
    #[test]
    fn a_backward_permute_reads_the_lane_its_address_names() {
        // v_mov_b32 v2, 11; ds_bpermute_b32 v1, v0, v2
        stores_v1(&[0x7E04_028B, 0xDACC_0000, 0x0100_0200]).unwrap();
    }

    /// A carry-in reading a join of compares after a divergent branch. A
    /// lane that skipped the inner compare has VCC clear there (VOPC clears
    /// inactive lanes), not the first compare's bit, so the join is no
    /// per-lane phi: lowering refuses it.
    #[test]
    fn a_carry_in_from_a_divergent_join_is_refused() {
        let body = [
            0x7E04_028B, // v_mov_b32 v2, 11
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE82_376A, // s_andn1_saveexec_b64 s[2:3], vcc
            0xBF88_0001, // s_cbranch_execz 1
            0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0x5002_0480, // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
        ];
        let e = stores_v1(&body).unwrap_err().to_string();
        assert!(e.contains("a carry-in from an untracked mask"), "{e}");
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
            group_split: 1,
            half_views: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
                group_split: 1,
                half_views: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// The LLVM text `dxc -dumpbin` prints for a DXIL container, when dxc
    /// is installed.
    #[cfg(windows)]
    fn dump_dxil(container: &[u8]) -> Option<String> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => std::path::PathBuf::from(p),
            None => super::validator::Validator::default_path()?.with_file_name("dxc.exe"),
        };
        if !dxc.exists() {
            return None;
        }
        let file = std::env::temp_dir().join(format!("victory-interp-{}.dxil", std::process::id()));
        std::fs::write(&file, container).ok()?;
        let out = std::process::Command::new(dxc)
            .arg("-dumpbin")
            .arg(&file)
            .output()
            .ok()?;
        let _ = std::fs::remove_file(&file);
        assert!(
            out.status.success(),
            "dxc -dumpbin failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[cfg(windows)]
    #[test]
    fn interpolated_f16_writes_the_op_sel_half_with_modifiers() {
        // Seed v2 with a non-constant value (root constant s0) so bitcode::opt
        // cannot fold away the mask keeping the untouched half; run p1_f32 into
        // v3 and complete it with p2_f16 into v2.
        let p2 = |dst_high: bool, omod: u32, clamp: bool| {
            let vgpr = |n: u32| 256 + n;
            let mut w0 = 0x35u32 << 26 | 0x35au32 << 16 | 2;
            if dst_high {
                w0 |= 1 << 14; // op_sel[3]: the high destination half.
            }
            if clamp {
                w0 |= 1 << 15;
            }
            let w1 = vgpr(1) << 9 | vgpr(3) << 18;
            [w0, w1 | omod << 27]
        };
        let dump = |dst_high: bool, omod: u32, clamp: bool| {
            let [w0, w1] = p2(dst_high, omod, clamp);
            let words = [
                0x7E04_0200u32, // v_mov_b32 v2, s0
                0xC80C_0000,    // v_interp_p1_f32 v3, v0, attr0.x
                w0,
                w1, // v_interp_p2_f16 v2, attr0, v1, v3
                0xF800_180F,
                0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
                0xBF81_0000,
            ];
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![crate::native::RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: true,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr: 2,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage, half| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0, true)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15, false)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::emit(&c, &native, &p, &st, &io, &res).unwrap().dxil
        };
        let (Some(low), Some(high)) = (
            dump_dxil(&dump(false, 0, false)),
            dump_dxil(&dump(true, 1, true)),
        ) else {
            eprintln!("no dxc.exe; DXIL check skipped");
            return;
        };
        // The low-half write keeps the destination's high half.
        assert!(low.contains("-65536"), "low write keeps the high half");
        // The high-half write shifts the result up by 16.
        assert!(high.contains("shl i32"), "op_sel[3] writes the high half");
        // omod mul:2 and clamp (dx.op.unary.f32, opcode 7) are applied.
        assert!(high.contains("2.000000e+00"), "omod multiplies by two");
        assert!(
            high.contains("dx.op.unary.f32(i32 7"),
            "clamp saturates the result"
        );
        assert!(!low.contains("dx.op.unary.f32(i32 7"));
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    #[test]
    fn valu_div_f32_sequence_executes_on_gpu() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x340E_0083, // v_lshlrev_b32 v7, 3, v0
            0xE034_1000,
            0x8002_0107, // buffer_load_dwordx2 v[1:2], v7, s[8:11], 0 offen
            0xD56D_6A03,
            0x0406_0502, // v_div_scale_f32 v3, vcc_lo, v2, v2, v1
            0xD56D_1008,
            0x0406_0501, // v_div_scale_f32 v8, s16, v1, v2, v1
            0x7E08_5503, // v_rcp_f32 v4, v3
            0xD54B_0005,
            0x23CA_0903, // v_fma_f32 v5, -v3, v4, 1.0
            0xD54B_0004,
            0x0412_0905, // v_fma_f32 v4, v5, v4, v4
            0x100A_0908, // v_mul_f32 v5, v8, v4
            0xD54B_0006,
            0x2422_0B03, // v_fma_f32 v6, -v3, v5, v8
            0xD56F_0004,
            0x0416_0906, // v_div_fmas_f32 v4, v6, v4, v5
            0xD55F_0001,
            0x0406_0504, // v_div_fixup_f32 v1, v4, v2, v1
            0x340E_0082, // v_lshlrev_b32 v7, 2, v0
            0xE070_1000,
            0x8003_0107, // buffer_store_dword v1, v7, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 9,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                crate::UsedResource {
                    kind: crate::ResourceType::Srv,
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                crate::UsedResource {
                    kind: crate::ResourceType::Uav,
                    root: crate::RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [7, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
        })
        .unwrap();
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Ok(host) = super::host::Host::new(false) {
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
                group_split: 1,
                half_views: l.half_views,
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let inputs: Vec<u32> = [
                7.0f32, 3.0, -1.0, 1e-40, 1.0, 1e35, 1e-35, 1e-36, 0.0, 0.0, -1.0, 0.0, -1e35, 1e35,
            ]
            .into_iter()
            .map(f32::to_bits)
            .collect();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(0, super::host::Root::Constants(vec![0, 0, 0, 0, 0]))],
                        views: &[
                            super::host::View::Buffer {
                                data: &inputs,
                                stride: super::host::Stride::Raw,
                            },
                            super::host::View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let got: Vec<f32> = out[0][..7].iter().copied().map(f32::from_bits).collect();
            assert!((got[0] - (7.0f32 / 3.0)).abs() < 1e-6, "got {:?}", got);
            assert_eq!(out[0][1], (-1.0f32 / 1e-40).to_bits());
            assert!(
                (got[2] - 1e-35).abs() <= 1e-40,
                "lane 2 got {:?} ({:#x})",
                got[2],
                out[0][2]
            );
            assert!(
                (got[3] - 10.0).abs() < 1e-5,
                "lane 3 got {:?} ({:#x})",
                got[3],
                out[0][3]
            );
            assert_eq!(out[0][4], 0xFFC0_0000);
            assert_eq!(out[0][5], (-1.0f32 / 0.0).to_bits());
            assert!(
                (got[6] - (-1.0)).abs() < 1e-6,
                "lane 6 got {:?} ({:#x})",
                got[6],
                out[0][6]
            );
        }
    }

    /// A pixel program the console tests depth for before it runs
    /// (`DB_SHADER_CONTROL` 0x1050: DEPTH_BEFORE_SHADER, early Z, kill, as
    /// the title's glass prepass has it) declares ForceEarlyDepthStencil:
    /// the module differs from the same program without the bit only by that
    /// flag, and the validator, where there is one, accepts it. With a depth
    /// output the bit is not declared (the host cannot test before a shader
    /// that writes the depth).
    #[test]
    fn a_pixel_program_tested_before_it_runs_declares_early_depth() {
        // v_mov_b32 v0, 1.0; exp mrt0 v0, v0, v0, v0 done vm; s_endpgm
        let color = [0x7E00_02F2u32, 0xF800_180F, 0x0000_0000, 0xBF81_0000];
        // v_mov_b32 v0, 1.0; exp mrtz v0, off, off, off done vm; s_endpgm
        let depth = [0x7E00_02F2u32, 0xF800_1881, 0x0000_0000, 0xBF81_0000];
        let el = |name: &str, kind, cols| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            // A render target has a register; depth and stencil do not.
            row: (kind == 16).then_some(0),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        };
        let emitted = |words: &[u32], out: SigElement, db: u32| {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 1,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: db,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let osg1 = container::signature(&[out], false);
            let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
            let c = Container::parse(&bytes).unwrap();
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            emit(&c, &native, &p, &st, &io, &res).unwrap().dxil
        };
        let late = emitted(&color, el("SV_Target", 16, 4), 0x50);
        let early = emitted(&color, el("SV_Target", 16, 4), 0x1050);
        assert_ne!(late, early, "the early-depth flag changed nothing");
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut early.clone()).unwrap();
        }
        let late = emitted(&depth, el("SV_Depth", 17, 1), 0x51);
        let early = emitted(&depth, el("SV_Depth", 17, 1), 0x1051);
        assert_eq!(late, early, "early depth declared for a depth export");
        // The rule alone: the bit, and no export of depth, stencil or coverage.
        let ps = |db| StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: db,
            patches: vec![],
        };
        assert!(lower::early_depth(&ps(0x1450), &[]));
        assert!(lower::early_depth(&ps(0x1010), &[]));
        assert!(!lower::early_depth(&ps(0x0450), &[]));
        assert!(!lower::early_depth(&ps(0x1012), &[]));
        assert!(!lower::early_depth(&ps(0x1150), &[]));
        assert!(!lower::early_depth(&ps(0x1050), &[el("SV_StencilRef", 20, 1)]));
        assert!(!lower::early_depth(&StageIo::None, &[]));
    }

    /// A program that reads its thread-group id in x takes the dispatch's
    /// first group from a user-data dword the host sets (`USER_GROUP_BASE`)
    /// and adds the host's group id to it: the host runs the groups past
    /// 65,535 of an indirect record as a second execution with it at 65,535.
    #[test]
    fn the_group_id_in_x_starts_at_the_host_base() {
        let words: &[u32] = &[
            0x7E02_0205, // v_mov_b32 v1, s5 (the group id in x)
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0302,
            0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], 0x0
            0x3610_028F, // v_and_b32 v8, 15, v1
            0x340E_1082, // v_lshlrev_b32 v7, 2, v8
            0xE070_1000,
            0x8003_0107, // buffer_store_dword v1, v7, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 9,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![crate::UsedResource {
                kind: crate::ResourceType::Uav,
                root: crate::RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            // Five user SGPRs, then the group id in x (s5).
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1 | 1 << 7,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
        })
        .unwrap();
        let base = l
            .user_data
            .iter()
            .find(|u| u.root == crate::RootType::Other(super::USER_GROUP_BASE))
            .map(|u| u.index as usize)
            .expect("no user-data dword for the first group");
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Ok(host) = super::host::Host::new(false) {
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
                group_split: 1,
                half_views: l.half_views,
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            // Three groups, with the first at 0 and at 65,535: each stores
            // the id it read at slot (id & 15).
            for (first, slots) in [(0u32, [0usize, 1, 2]), (65535, [15, 0, 1])] {
                let mut constants = vec![0u32; base + 1];
                constants[base] = first;
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &super::host::Dispatch {
                            roots: vec![(0, super::host::Root::Constants(constants))],
                            views: &[super::host::View::RawUav(64)],
                            samplers: &[],
                            groups: [3, 1, 1],
                        },
                    )
                    .unwrap();
                for (k, slot) in slots.into_iter().enumerate() {
                    assert_eq!(
                        out[0][slot],
                        first + k as u32,
                        "group {k} with the first at {first}: {:?}",
                        &out[0][..16]
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use crate::native::{ComputeIo, RootParam, StageIo};

    #[test]
    fn heap_texture_atomic_widths() {
        // A synthetic compute shader loads a 2D UAV from a root table,
        // compares or adds at (0, 0), then uses the returned register(s).
        for (opcode, mask, wide) in [
            (16u32, 3, false),
            (16, 15, true),
            (17, 1, false),
            (17, 3, true),
        ] {
            let words = [
                0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
                0xF40C_0200,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
                0x7E00_0280,
                0x7E02_0280, // v0, v1 = coordinates
                0x7E04_0281,
                0x7E06_0282,
                0x7E08_0283,
                0x7E0A_0284, // v2..v5 = 1..4
                0xF000_2008 | opcode << 18 | mask << 8,
                0x0002_0200,
                // Use the old value in a second atomic at the same location.
                0xF044_0008 | (if wide { 3 } else { 1 }) << 8,
                0x0002_0200,
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 6,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let bytes = crate::dxbc::build(&[]);
            let shader = emit(
                &Container::parse(&bytes).unwrap(),
                &native,
                &p,
                &structure::structure(&p).unwrap(),
                &io::raise(&p, &native).unwrap(),
                &resource::track(&p, &native.layout, native.stage, None).unwrap(),
            )
            .unwrap();
            let container = Container::parse(&shader.dxil).unwrap();
            let features = u64::from_le_bytes(
                container
                    .part(FourCC::SFI0)
                    .unwrap()
                    .data
                    .try_into()
                    .unwrap(),
            );
            let required = 0x1040_8000; // heap atomics, typed atomics, Int64Ops
            assert_eq!(features & required, if wide { required } else { 0 });
            #[cfg(windows)]
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        pixel_with(words, input_addr, false)
    }

    /// `pixel`, with TEXCOORD0 held as 16-bit values when `fp16`.
    fn pixel_with(words: &[u32], input_addr: u32, fp16: bool) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(&[element("TEXCOORD", 0, 0)], true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let res = resource::track(&p, &native.layout, native.stage, None)?;
        let io = io::raise(&p, &native)?;
        let st = structure::structure(&p)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// An f16 interpolation's first half keeps no value of its own (the second half writes
    /// another VGPR and recomputes it). A `v_fma_mixlo_f16` into that VGPR keeps its high
    /// half: zero then, as for other 16-bit writes, and the program translates.
    /// Eight of the title's deferred material pixel programs do this.
    #[test]
    fn mixlo_over_an_f16_first_interpolation_half() {
        let [e0, e1] = exp_mrt0(4);
        let words = [
            0xD742_0003, 0x0002_0000, // v_interp_p1ll_f16 v3, v0, attr0.x
            0xD75A_0002, 0x040E_0200, // v_interp_p2_f16 v2, v1, attr0.x, v3
            0xCC21_0003, 0x0A02_0502, // v_fma_mixlo_f16 v3, v2, v2, 0
            0x7E08_1703,              // v_cvt_f32_f16 v4, v3
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel_with(&words, PERSP_CENTER, true).unwrap();
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }

    fn lower_ps(words: &[u32]) -> lower::Lowered {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: PERSP_CENTER,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), false, 8).unwrap();
        let res = resource::track(&p, &[], HwStage::Ps, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            view_marks: false,
        })
        .unwrap()
    }

    #[test]
    fn pixel_readfirstlane_uses_is_helper_lane_and_compute_does_not() {
        let [e0, e1] = exp_mrt0(3);
        let ps_words = [
            0x7E04_0500, // v_readfirstlane_b32 s2, v0
            0x7E06_0202, // v_mov_b32 v3, s2
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&ps_words, PERSP_CENTER).unwrap();
        validate(&shader);

        let lowered = lower_ps(&ps_words);
        assert!(
            lowered.funcs.iter().any(|n| n == "dx.op.isHelperLane.i1"),
            "pixel shader must guard v_readfirstlane with isHelperLane: {:?}",
            lowered.funcs
        );

        use crate::native::{ComputeIo, ResourceType, RootParam, RootType, UsedResource};
        let cs_words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
            125 << 25,   // s_load_dwordx4 s[8:11], s[4:5], 0
            0x7E04_0500, // v_readfirstlane_b32 s2, v0
            0x7E02_0202, // v_mov_b32 v1, s2
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16 | 1 << 8, // buffer_store_dword v1, off, s[8:11], 0
            S_ENDPGM,
        ];
        let cs_native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [32, 1, 1],
            }),
            code: cs_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&cs_native.patched_code(), false, 8).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &cs_native.layout, HwStage::Cs, Some(&st)).unwrap();
        let io = io::raise(&p, &cs_native).unwrap();
        let lowered = lower::lower(&lower::Parts {
            native: &cs_native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            view_marks: false,
        })
        .unwrap();
        assert!(
            !lowered.funcs.iter().any(|n| n == "dx.op.isHelperLane.i1"),
            "compute shader must not emit isHelperLane"
        );
    }

    #[test]
    fn pixel_quad_swizzle_without_discards_uses_quad_read_lane_at() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xD8D4_80E4,
            0x0300_0000, // ds_swizzle_b32 v3, v0 offset:0x80E4
            v_add_f32(2, 3, 1),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        validate(&shader);

        let lowered = lower_ps(&words);
        assert!(lowered.funcs.iter().any(|n| n == "dx.op.quadReadLaneAt.i32"));
        assert!(!lowered.funcs.iter().any(|n| n == "dx.op.waveReadLaneAt.i32"));
    }

    /// Writes through unordered buffer views mark their 64 KiB pages when
    /// `view_marks` is enabled: a dword store and a 16-byte store through
    /// view 0 (base bit 100, phase 0xFFF0 so the 16-byte store at offset 12
    /// crosses into page 101), a store through view 1 (VIEW_MARK_NONE), and
    /// an atomic add through view 2 (base bit 200). With `view_marks: false`,
    /// no view-mark root bindings are emitted.
    #[test]
    fn writes_through_views_mark_their_pages() {
        use crate::native::{ComputeIo, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};

        const fn s_mov_b32(dst: u8, src: u8) -> u32 {
            0xBE80_0300 | (dst as u32) << 16 | src as u32
        }
        const fn s_load_dwordx4(sbase: u8, sdata: u8, offset: u32) -> [u32; 2] {
            [
                0x3D << 26 | 2 << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
                (0x7D << 25) | offset,
            ]
        }
        const fn v_mov_b32(vdst: u8, src: u8) -> u32 {
            0x7E00_0200 | (vdst as u32) << 17 | src as u32
        }
        const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
            [
                0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
            ]
        }
        const BUFFER_STORE_DWORD: u32 = 0x1C;
        const BUFFER_STORE_DWORDX4: u32 = 0x1E;
        const BUFFER_ATOMIC_ADD: u32 = 0x32;
        const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
        const ZERO: u8 = 0x80;

        let mut w = vec![s_mov_b32(5, ZERO)];
        for (k, sdata) in [8u8, 12, 16].into_iter().enumerate() {
            w.extend(s_load_dwordx4(4, sdata, 32 * k as u32));
        }
        w.push(S_WAITCNT_ALL);
        w.extend((0..4).map(|v| v_mov_b32(v, 0)));
        w.extend(mubuf(BUFFER_STORE_DWORD, 0, 8, 0));
        w.extend(mubuf(BUFFER_STORE_DWORDX4, 0, 8, 12));
        w.extend(mubuf(BUFFER_STORE_DWORD, 0, 12, 4));
        w.extend(mubuf(BUFFER_ATOMIC_ADD, 0, 16, 8));
        w.push(S_ENDPGM);

        let param = |root, user_data, sgprs, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let uav = |k: u32| UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 8 * k,
            lower_bound: k,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = NativeProgram {
            stage: crate::HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                param(RootType::Constants, 0, 4, 0),
                param(RootType::Table, 4, 1, 1),
            ],
            used: (0..3).map(uav).collect(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = crate::structure::structure(&p).expect("structure");
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = crate::io::raise(&p, &native).expect("io::raise");

        // Disabled: zero view-mark bindings.
        let off = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
        })
        .expect("lower off");
        assert!(!off.root_srvs.contains(&super::SLOT_VIEW_MARKS));
        assert!(!off.root_uavs.contains(&super::SLOT_VIEW_WRITES));

        // Enabled: emits both bindings, validates, and sets exact bits on GPU.
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: true,
        })
        .expect("lower on");
        assert!(l.root_srvs.contains(&super::SLOT_VIEW_MARKS));
        assert!(l.root_uavs.contains(&super::SLOT_VIEW_WRITES));
        let shader = super::HostShader {
            stage: crate::HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: res.root_indexed_views,
            half_views: l.half_views,
            group_split: 1,
        };
        assert!(shader.has_view_marks());
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        #[cfg(windows)]
        {
            use super::host::{Dispatch, Host, Root, View};
            let _one = super::valu_tests::ONE_AT_A_TIME
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let value = 0x1234_5678u32;
            let marks = vec![100, 0xFFF0, super::VIEW_MARK_NONE, 0, 200, 0];
            let mut ran = 0;
            for (label, warp) in [("hardware", false), ("warp", true)] {
                let Ok(host) = Host::new(warp) else {
                    eprintln!("no SM 6.6 {label} device; skipped");
                    continue;
                };
                let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
                let mut ud = vec![0u32; shader.user_data_dwords() as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = if u.root == RootType::Constants { value } else { 0 };
                }
                // `root_signature_for` sorts root views by slot:
                // param 0 = user data, param 1 = SLOT_VIEW_WRITES (250), param 2 = SLOT_VIEW_MARKS (251).
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![
                                (0, Root::Constants(ud)),
                                (1, Root::Uav(8)),
                                (2, Root::View { srv: true, data: marks.clone() }),
                            ],
                            views: &[View::RawUav(8), View::RawUav(8), View::RawUav(8)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(out[0], [value, 0, 0, value, value, value, value, 0], "{label}: view 0");
                assert_eq!(out[1][1], value, "{label}: view 1");
                assert_eq!(out[2][2], value, "{label}: atomic");
                let marked: Vec<usize> = (0..out[3].len() * 32)
                    .filter(|&b| out[3][b / 32] >> (b % 32) & 1 != 0)
                    .collect();
                assert_eq!(marked, [100, 101, 200], "{label}: marked pages");
                ran += 1;
            }
            assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
        }
    }

    #[test]
    fn checked_single_wave_compute_hashes_enable_wide_unless_opted_out() {
        use std::ffi::OsStr;

        let checked = [
            "6871d6e4b3ac9a10",
            "926c46bab3b86bd6",
            "a10179ab7eda0dc3",
            "369cbe906ea72764",
            "2cc4c197ee29e291",
        ];
        for hash in checked {
            assert!(
                wide_lowering_enabled(hash, None, &[], false, 2),
                "{hash} with group_sync=2"
            );
            assert!(
                wide_lowering_enabled(hash, None, &[], false, 1),
                "{hash} with group_sync=1"
            );
            assert!(
                !wide_lowering_enabled(hash, None, &[], false, 0),
                "{hash} must stay off when group_sync=0"
            );
            assert!(
                !wide_lowering_enabled(hash, Some(OsStr::new("0")), &[], false, 2),
                "{hash} must opt out with VICTORY_WAVE_WIDE=0"
            );
            assert!(
                wide_lowering_enabled(hash, Some(OsStr::new("1")), &[], false, 0),
                "{hash} with explicit VICTORY_WAVE_WIDE=1"
            );
        }

        // Explicitly excluded hashes that caused black rectangles on water must be false.
        for excluded in ["5dc3b25986622949", "6f225b641c5b0708"] {
            assert!(
                !wide_lowering_enabled(excluded, None, &[], false, 2),
                "{excluded} must be excluded from default wide allowlist"
            );
            assert!(
                !wide_lowering_enabled(excluded, None, &[], false, 1),
                "{excluded} must stay off with group_sync=1"
            );
        }

        for other in ["", "f133e35efed2a48a", "a53a7977fe762268", "0123456789abcdef"] {
            assert!(
                !wide_lowering_enabled(other, None, &[], false, 2),
                "{other} must stay off by default"
            );
            assert!(
                !wide_lowering_enabled(other, Some(OsStr::new("0")), &[], false, 2),
                "{other} with VICTORY_WAVE_WIDE=0"
            );
            assert!(
                wide_lowering_enabled(other, Some(OsStr::new("1")), &[], false, 2),
                "{other} with explicit VICTORY_WAVE_WIDE=1"
            );
        }
    }

    #[test]
    fn wave_wide_only_overrides_default_allowlist_and_matches_prefixes() {
        use std::ffi::OsStr;

        let only = parse_wave_wide_only(Some(" 6871D6E4 , deadbeef , "));
        assert_eq!(only, vec!["6871d6e4".to_owned(), "deadbeef".to_owned()]);
        assert!(matches_wave_wide_only("6871d6e4b3ac9a10", &only));
        assert!(matches_wave_wide_only("DEADBEEF00112233", &only));
        assert!(!matches_wave_wide_only("926c46bab3b86bd6", &only));
        assert!(!matches_wave_wide_only("", &only));

        // When VICTORY_WAVE_WIDE_ONLY is set (non-empty), ONLY matching hashes (or WIDE_HERE /
        // explicit VICTORY_WAVE_WIDE=1) enable wide lowering; other default hashes stay off.
        assert!(wide_lowering_enabled("6871d6e4b3ac9a10", None, &only, false, 2));
        assert!(wide_lowering_enabled("deadbeef12345678", None, &only, false, 2));
        assert!(!wide_lowering_enabled("926c46bab3b86bd6", None, &only, false, 2));
        assert!(!wide_lowering_enabled("369cbe906ea72764", None, &only, false, 2));
        assert!(!wide_lowering_enabled("2cc4c197ee29e291", None, &only, false, 2));
        assert!(wide_lowering_enabled("", None, &only, true, 2));
        assert!(!wide_lowering_enabled("", None, &only, false, 2));
        assert!(wide_lowering_enabled("926c46bab3b86bd6", Some(OsStr::new("1")), &only, false, 2));
        assert!(!wide_lowering_enabled("6871d6e4b3ac9a10", Some(OsStr::new("0")), &only, true, 2));

        // Thread-local `with_wave_wide` scopes and restores its previous state.
        assert!(!WIDE_HERE.get());
        with_wave_wide(true, || {
            assert!(WIDE_HERE.get());
            assert!(wide_lowering_enabled("0123456789abcdef", None, &[], WIDE_HERE.get(), 2));
            with_wave_wide(false, || {
                assert!(!WIDE_HERE.get());
            });
            assert!(WIDE_HERE.get());
        });
        assert!(!WIDE_HERE.get());
    }
}
