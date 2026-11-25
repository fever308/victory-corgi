//! A stand-in for ray-traced reflections while the structures they walk
//! are not built.
//!
//! The title's specular reflections are ray traced (`sr_tracing:*`): rays
//! walk the ray-tracing structures (`CS_RaytraceVBuffer_*`), the hits are
//! shaded by about 60 per-material `<material>:CS_RaytracingResolve`
//! programs, the misses get sky and probe radiance from `CS_RayMiss`
//! (06bb6e728c7c99bf), and `CS_ResolveScreenSpace` adds what the screen
//! itself shows; all of them write one half-resolution radiance image
//! (R11G11B10, table UAV `s1+104` of `CS_RayMiss`), which the denoiser
//! (`sr_denoising:*`) then blurs and keeps a history of. The structure
//! builder does not run here (see `crate::pipeline::builder_runs`): every
//! ray walks whatever bytes lie where the structures would be, so what the
//! resolves and `CS_RayMiss` leave is mostly sky, even on eyes in shadow,
//! inside mouths, on skin.
//!
//! With `VICTORY_REFLECTION`, while the builder has not run whole, that
//! image is cleared right after `CS_RayMiss` (which runs after the
//! resolves, before `CS_ResolveScreenSpace`): what the structures' rays
//! brought is dropped, the screen-space reflections that follow stay.
//! Values, separated by commas:
//! - unset, empty, `0` or `off`: off (the default);
//! - `miss`: cleared to zero;
//! - a positive number: cleared to that grey level (linear radiance);
//! - `alpha`, `water`: the images of the transparent and water passes too
//!   (told apart by their sky-probe fallbacks, `CS_DeferredFallbacksAlpha_
//!   SkyProbe` b7953a05b189dfc1 and `CS_DeferredFallbacksWater_SkyProbe`
//!   8673201660135e58, which write them before; left as the title has them
//!   unless asked for);
//! - `history`: the outputs of the denoiser's last passes
//!   (`CS_BilateralBlur` dce085c07b9414b6, `CS_UpdateHistory`
//!   6fad2980c2774c90, `CS_BilateralBlurAndUpdateHistory` 2d16a9b30ad8b6bb)
//!   cleared to zero after them: no ray-traced specular at all, also none
//!   from the screen.
//!
//! The image is found as `ambient.rs` of fxceless/victory-fork pull
//! request 917 (VICTORY_AMBIENT, after Belobug's VIctory-Belobug-AMD)
//! finds its own: the program known at its dispatch, the view its table
//! names at the place its code reads. It is cleared with a render target
//! clear right after the dispatch in the same list, after a transition the
//! list records (it waits for the pass's writes); the next use moves it
//! back.
//!
//! shortcut: the same darkness for every glossy surface, whatever the
//! scene; of no use once the builder runs (it stops acting then).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use crate::host::HostList;
use crate::object::This;

/// `sr_tracing:CS_RayMiss`.
const RAY_MISS: &str = "06bb6e728c7c99bf";
/// Where it writes the radiance image: root parameter and place in its
/// table, in dwords (`s1+104` in the frame listings).
const RADIANCE: (u8, u32) = (1, 104);
/// The denoiser's last passes, whose outputs `history` clears.
const DENOISED: [&str; 3] = [
    "dce085c07b9414b6", // sr_denoising:CS_BilateralBlur
    "6fad2980c2774c90", // sr_denoising:CS_UpdateHistory
    "2d16a9b30ad8b6bb", // sr_denoising:CS_BilateralBlurAndUpdateHistory
];
/// The sky-probe fallbacks of the transparent and water passes: what they
/// write is that pass's radiance image.
const FALLBACK_ALPHA: &str = "b7953a05b189dfc1";
const FALLBACK_WATER: &str = "8673201660135e58";

/// What `VICTORY_REFLECTION` asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Mode {
    /// The level `CS_RayMiss`'s image is cleared to (none: not cleared).
    miss: Option<f32>,
    /// Also the transparent pass's image.
    alpha: bool,
    /// Also the water pass's image.
    water: bool,
    /// The denoiser's outputs, to zero.
    history: bool,
}

/// Which radiance image a pass's is, told by the fallback that wrote it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Alpha,
    Water,
}

/// What a dispatch of a program is to this pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    /// `CS_RayMiss`: its image is cleared.
    Miss,
    /// One of the denoiser's last passes: its outputs are cleared.
    Denoised,
    /// A sky-probe fallback: its images are of that kind.
    Fallback(Kind),
}

/// The mode `set` asks for: none if off, the token not understood if any.
fn mode_of(set: &str) -> Result<Option<Mode>, String> {
    let set = set.trim().to_ascii_lowercase();
    if matches!(set.as_str(), "" | "0" | "off" | "false" | "no") {
        return Ok(None);
    }
    let mut mode = Mode {
        miss: None,
        alpha: false,
        water: false,
        history: false,
    };
    let mut level = None;
    for token in set.split([',', '+']).map(str::trim).filter(|t| !t.is_empty()) {
        match token {
            "miss" => mode.miss = Some(0.0),
            "alpha" => mode.alpha = true,
            "water" => mode.water = true,
            "history" => mode.history = true,
            _ => match token.parse::<f32>() {
                Ok(l) if l.is_finite() && l > 0.0 => level = Some(l),
                _ => return Err(token.to_owned()),
            },
        }
    }
    if level.is_some() {
        mode.miss = level;
    }
    if (mode.alpha || mode.water) && mode.miss.is_none() {
        mode.miss = Some(0.0);
    }
    Ok((mode.miss.is_some() || mode.history).then_some(mode))
}

/// What program `hash` (as logged) is to `mode`.
fn pass_of(hash: &str, mode: &Mode) -> Option<Pass> {
    let miss = mode.miss.is_some();
    match hash {
        RAY_MISS if miss => Some(Pass::Miss),
        FALLBACK_ALPHA if miss => Some(Pass::Fallback(Kind::Alpha)),
        FALLBACK_WATER if miss => Some(Pass::Fallback(Kind::Water)),
        h if mode.history && DENOISED.contains(&h) => Some(Pass::Denoised),
        _ => None,
    }
}

/// Whether `CS_RayMiss`'s image of `kind` (none: the opaque pass's) is
/// cleared in `mode`.
fn clears(kind: Option<Kind>, mode: &Mode) -> bool {
    match kind {
        None => true,
        Some(Kind::Alpha) => mode.alpha,
        Some(Kind::Water) => mode.water,
    }
}

/// Of the table UAVs a program names (root parameter, place in dwords),
/// the one that is `CS_RayMiss`'s radiance image: at its known place, or
/// the only one.
fn radiance_of(uavs: &[(u8, u32)]) -> Option<usize> {
    uavs.iter().position(|&u| u == RADIANCE).or((uavs.len() == 1).then_some(0))
}

/// The colour a render target clear of `format` (DXGI) to `level` takes;
/// none if the format is not one this pass clears (typeless, packed
/// shared-exponent, depth), or an integer format asked for other than zero.
fn colour_of(format: u32, level: f32) -> Option<[f32; 4]> {
    const FLOAT: [u32; 8] = [
        2,  // R32G32B32A32_FLOAT
        10, // R16G16B16A16_FLOAT
        24, // R10G10B10A2_UNORM
        26, // R11G11B10_FLOAT
        28, // R8G8B8A8_UNORM
        34, // R16G16_FLOAT
        41, // R32_FLOAT
        54, // R16_FLOAT
    ];
    const INTEGER: [u32; 2] = [
        42, // R32_UINT
        57, // R16_UINT
    ];
    if FLOAT.contains(&format) {
        Some([level, level, level, 1.0])
    } else if INTEGER.contains(&format) && level == 0.0 {
        Some([0.0; 4])
    } else {
        None
    }
}

/// What a title's unordered access view `desc` (D3D12 words: format,
/// dimension, then the dimension's fields) of a texture with `format` and
/// `slices` covers: its format, mip level, first slice and slice count;
/// none if it is not of a 2D texture.
fn extent_of(desc: Option<[u32; 10]>, format: u32, slices: u32) -> Option<(u32, u32, u32, u32)> {
    const TEXTURE2D: u32 = 4;
    const TEXTURE2DARRAY: u32 = 5;
    let Some(d) = desc else {
        return Some((format, 0, 0, slices));
    };
    let format = if d[0] != 0 { d[0] } else { format };
    match d[1] {
        TEXTURE2D => Some((format, d[2], 0, 1)),
        TEXTURE2DARRAY => {
            let count = if d[4] == u32::MAX || d[4] == 0 { slices.saturating_sub(d[3]) } else { d[4] };
            Some((format, d[2], d[3], count.max(1)))
        }
        _ => None,
    }
}

/// The mode (`VICTORY_REFLECTION`), read once; none if off.
pub(crate) fn mode() -> Option<&'static Mode> {
    static MODE: OnceLock<Option<Mode>> = OnceLock::new();
    MODE.get_or_init(|| {
        let set = std::env::var("VICTORY_REFLECTION").unwrap_or_default();
        match mode_of(&set) {
            Ok(Some(mode)) => {
                tracing::info!(
                    set,
                    ?mode,
                    "ray-traced reflections are cleared while the ray-tracing structures are not built (VICTORY_REFLECTION)"
                );
                Some(mode)
            }
            Ok(None) => None,
            Err(token) => {
                tracing::warn!(set, token, "VICTORY_REFLECTION is not understood: it stays off");
                None
            }
        }
    })
    .as_ref()
}

/// What program `hash` is to the mode on, if it is on.
pub(crate) fn pass(hash: &str) -> Option<(Pass, &'static Mode)> {
    let mode = mode()?;
    pass_of(hash, mode).map(|p| (p, mode))
}

/// The radiance images of the transparent and water passes, by title
/// resource: kept for the run (the title keeps those images).
static KINDS: Mutex<Option<HashMap<usize, Kind>>> = Mutex::new(None);

/// A table UAV a dispatch names: root parameter, place in dwords, title
/// resource and view description.
pub(crate) type Uav = (u8, u32, usize, Option<[u32; 10]>);

/// After a dispatch of a program that is `pass`, recorded in `list`, whose
/// table UAVs are `uavs`: notes or clears what it wrote. Only while the
/// ray-tracing structure builder has not run whole.
pub(crate) fn after(list: &HostList, (pass, mode): (Pass, &Mode), hash: &str, uavs: &[Uav]) {
    if crate::pipeline::builder_whole() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::info!("the ray-tracing structures are built: VICTORY_REFLECTION no longer acts"));
        return;
    }
    match pass {
        Pass::Fallback(kind) => {
            let mut kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
            let kinds = kinds.get_or_insert_with(HashMap::new);
            for &(_, _, resource, _) in uavs {
                kinds.insert(resource, kind);
            }
        }
        Pass::Miss => {
            let places: Vec<(u8, u32)> = uavs.iter().map(|&(slot, at, ..)| (slot, at)).collect();
            let Some(i) = radiance_of(&places) else {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(hash, ?places, "no reflection clear: CS_RayMiss's tables are not as expected"));
                return;
            };
            let (_, _, resource, desc) = uavs[i];
            let kind = KINDS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .and_then(|k| k.get(&resource).copied());
            if !clears(kind, mode) {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        hash,
                        ?kind,
                        resource = format_args!("{resource:#x}"),
                        "a reflection image of the transparent or water pass is left as the title has it (VICTORY_REFLECTION without alpha/water)"
                    )
                });
                return;
            }
            let level = mode.miss.unwrap_or(0.0);
            static LOGGED: [std::sync::Once; 3] = [const { std::sync::Once::new() }; 3];
            let logged = &LOGGED[match kind {
                None => 0,
                Some(Kind::Alpha) => 1,
                Some(Kind::Water) => 2,
            }];
            clear(list, (resource, desc), level, hash, kind, logged);
        }
        Pass::Denoised => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            let mut done: Vec<usize> = Vec::new();
            for &(_, _, resource, desc) in uavs {
                if done.contains(&resource) {
                    continue;
                }
                done.push(resource);
                clear(list, (resource, desc), 0.0, hash, None, &LOGGED);
            }
        }
    }
}

/// Clears title texture `resource` through the part of it `desc` names to
/// `level`, in `list`; logs the first time `logged` is.
fn clear(
    list: &HostList,
    (resource, desc): (usize, Option<[u32; 10]>),
    level: f32,
    hash: &str,
    kind: Option<Kind>,
    logged: &std::sync::Once,
) -> bool {
    const RENDER_TARGET: u32 = 0x4;
    let Some(host) = crate::host::host() else {
        return false;
    };
    let Some(texture) = crate::resource::host_resource(resource as This) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let extent = (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 != 0
        && d.SampleDesc.Count == 1)
        .then(|| extent_of(desc, d.Format.0 as u32, u32::from(d.DepthOrArraySize)))
        .flatten();
    let colour = extent.and_then(|(format, ..)| colour_of(format, level));
    let (Some((format, mip, first, count)), Some(colour)) = (extent, colour) else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                hash,
                resource = format_args!("{resource:#x}"),
                dimension = d.Dimension.0,
                flags = d.Flags.0,
                format = d.Format.0,
                ?desc,
                level,
                "no reflection clear: the image is not a render-target 2D texture of a format this clears"
            )
        });
        return false;
    };
    logged.call_once(|| {
        tracing::info!(
            hash,
            ?kind,
            resource = format_args!("{resource:#x}"),
            width = (d.Width >> mip).max(1),
            height = (d.Height >> mip).max(1),
            format,
            mip,
            slices = count,
            level,
            "a reflection image is cleared after its pass while the ray-tracing structures are not built (VICTORY_REFLECTION); the first one"
        )
    });
    let view = if u32::from(d.DepthOrArraySize) > 1 {
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT(format as i32),
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                    MipSlice: mip,
                    FirstArraySlice: first,
                    ArraySize: count,
                    PlaneSlice: 0,
                },
            },
        }
    } else {
        D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT(format as i32),
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: mip,
                    PlaneSlice: 0,
                },
            },
        }
    };
    // The pass wrote it as unordered access: the transition waits for it.
    crate::list::ensure_state(list, resource as This, RENDER_TARGET);
    let rtv = host.rtv(Some(texture), Some(&view));
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("reflection clear {resource:#x} f{format} mip {mip} to {level} after [compute {hash}] (VICTORY_REFLECTION)"));
    // SAFETY: an open list; a fresh view of a live texture in the render
    // target state, which the list keeps as the title's resource.
    unsafe { list.list.ClearRenderTargetView(rtv, &colour, None) };
    list.mark_work();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(miss: Option<f32>, alpha: bool, water: bool, history: bool) -> Option<Mode> {
        Some(Mode {
            miss,
            alpha,
            water,
            history,
        })
    }

    /// Off unless asked for; what each value asks for.
    #[test]
    fn the_switch_is_read() {
        for off in ["", " ", "0", "off", "OFF", "false", "no"] {
            assert_eq!(mode_of(off), Ok(None), "{off:?}");
        }
        assert_eq!(mode_of("miss"), Ok(on(Some(0.0), false, false, false)));
        assert_eq!(mode_of(" 0.05 "), Ok(on(Some(0.05), false, false, false)));
        assert_eq!(mode_of("miss,0.2"), Ok(on(Some(0.2), false, false, false)));
        assert_eq!(mode_of("history"), Ok(on(None, false, false, true)));
        assert_eq!(mode_of("miss,history"), Ok(on(Some(0.0), false, false, true)));
        assert_eq!(mode_of("miss+alpha"), Ok(on(Some(0.0), true, false, false)));
        assert_eq!(mode_of("water"), Ok(on(Some(0.0), false, true, false)));
        assert_eq!(mode_of("Miss, Alpha, Water"), Ok(on(Some(0.0), true, true, false)));
        assert_eq!(mode_of("hist"), Err("hist".to_owned()));
        assert_eq!(mode_of("-1"), Err("-1".to_owned()));
        assert_eq!(mode_of("nan"), Err("nan".to_owned()));
        assert_eq!(mode_of("inf"), Err("inf".to_owned()));
        assert_eq!(mode_of("miss,0"), Err("0".to_owned()));
    }

    /// The programs are known by their hashes, and only for what is on.
    #[test]
    fn the_passes_are_known_by_their_hashes() {
        let miss = on(Some(0.0), false, false, false).unwrap();
        let history = on(None, false, false, true).unwrap();
        assert_eq!(pass_of("06bb6e728c7c99bf", &miss), Some(Pass::Miss));
        assert_eq!(pass_of("b7953a05b189dfc1", &miss), Some(Pass::Fallback(Kind::Alpha)));
        assert_eq!(pass_of("8673201660135e58", &miss), Some(Pass::Fallback(Kind::Water)));
        assert_eq!(pass_of("dce085c07b9414b6", &miss), None);
        assert_eq!(pass_of("06bb6e728c7c99bf", &history), None);
        for h in ["dce085c07b9414b6", "6fad2980c2774c90", "2d16a9b30ad8b6bb"] {
            assert_eq!(pass_of(h, &history), Some(Pass::Denoised), "{h}");
        }
        // CS_RayMiss_Volumetrics, CS_ScreenProbe_RayMiss, CS_ResolveScreenSpace.
        for h in ["47ab71a31354a7c6", "b48d48f8304e5bd3", "130d3d04172e5456", ""] {
            assert_eq!(pass_of(h, &miss), None, "{h}");
            assert_eq!(pass_of(h, &history), None, "{h}");
        }
    }

    /// The opaque pass's image always; the others' only if asked for.
    #[test]
    fn transparent_and_water_images_are_asked_for() {
        let miss = on(Some(0.0), false, false, false).unwrap();
        let all = on(Some(0.0), true, true, false).unwrap();
        assert!(clears(None, &miss));
        assert!(!clears(Some(Kind::Alpha), &miss));
        assert!(!clears(Some(Kind::Water), &miss));
        assert!(clears(Some(Kind::Alpha), &all));
        assert!(clears(Some(Kind::Water), &all));
    }

    /// CS_RayMiss's image is at s1+104, or its only UAV.
    #[test]
    fn the_radiance_image_is_found() {
        assert_eq!(radiance_of(&[(1, 104)]), Some(0));
        assert_eq!(radiance_of(&[(1, 96), (1, 104)]), Some(1));
        assert_eq!(radiance_of(&[(1, 96)]), Some(0));
        assert_eq!(radiance_of(&[(1, 96), (1, 112)]), None);
        assert_eq!(radiance_of(&[]), None);
    }

    /// Float images to any level, integer ones only to zero, others never.
    #[test]
    fn clear_colours() {
        assert_eq!(colour_of(26, 0.0), Some([0.0, 0.0, 0.0, 1.0]));
        assert_eq!(colour_of(26, 0.1), Some([0.1, 0.1, 0.1, 1.0]));
        assert_eq!(colour_of(42, 0.0), Some([0.0; 4]));
        assert_eq!(colour_of(42, 0.1), None);
        assert_eq!(colour_of(67, 0.0), None); // R9G9B9E5_SHAREDEXP
        assert_eq!(colour_of(23, 0.0), None); // R10G10B10A2_TYPELESS
        assert_eq!(colour_of(0, 0.0), None); // UNKNOWN
    }

    /// What a view covers.
    #[test]
    fn view_extents() {
        assert_eq!(extent_of(None, 26, 1), Some((26, 0, 0, 1)));
        assert_eq!(extent_of(Some([0, 4, 0, 0, 0, 0, 0, 0, 0, 0]), 26, 1), Some((26, 0, 0, 1)));
        assert_eq!(extent_of(Some([42, 4, 3, 0, 0, 0, 0, 0, 0, 0]), 42, 1), Some((42, 3, 0, 1)));
        assert_eq!(extent_of(Some([26, 5, 0, 2, 3, 0, 0, 0, 0, 0]), 26, 8), Some((26, 0, 2, 3)));
        assert_eq!(extent_of(Some([26, 5, 1, 2, u32::MAX, 0, 0, 0, 0, 0]), 26, 8), Some((26, 1, 2, 6)));
        assert_eq!(extent_of(Some([26, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 26, 1), None);
        assert_eq!(extent_of(Some([26, 8, 0, 0, 0, 0, 0, 0, 0, 0]), 26, 1), None);
    }
}
