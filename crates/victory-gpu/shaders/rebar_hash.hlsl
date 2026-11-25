// 128-bit per-64-KiB-piece hash and compare pass for ReBAR UAV shadow buffers.
// Each 64-thread group hashes one 64 KiB piece using coalesced 16-byte loads
// (1024 bytes per stride, 64 iterations per full piece), then combines the 64
// lane hashes in groupshared memory with a non-commutative mixing tree.
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"

cbuffer Params : register(b0) {
    uint first_piece;  // first 64 KiB piece index in `src` and `hashes`
    uint aligned_len;  // buffer length rounded up to 16 bytes (<= allocated width)
    uint out_offset;   // byte offset in `flags` for group 0 (when mode != 0)
    uint mode;         // 0: store hashes; 1: compare against hashes, flag, and update
    uint pieces;       // number of contiguous pieces in this dispatch
    uint mark_base;    // base bit in `marks` (0xFFFFFFFFu = no view-mark filter)
    uint mark_skew;    // non-zero if shadow start is not 64 KiB aligned (piece spans 2 title pages)
};

ByteAddressBuffer src : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer flags : register(u1);
RWByteAddressBuffer marks : register(u2);

groupshared uint4 s_hash[64];

uint4 mix_pair(uint4 a, uint4 b) {
    uint4 x = (a ^ (b * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu)))
            * uint4(0x01000193u, 0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u);
    return x.yzwx ^ (x >> 15);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    if (gid.x >= pieces) return;

    uint tid = gtid.x;
    uint piece = first_piece + gid.x;

    bool active = true;
    if (mode != 0u && mark_base != 0xFFFFFFFFu) {
        uint bit0 = min(mark_base + piece, 4194303u);
        uint hit = (marks.Load((bit0 >> 5u) << 2u) >> (bit0 & 31u)) & 1u;
        if (hit == 0u && mark_skew != 0u) {
            uint bit1 = min(mark_base + piece + 1u, 4194303u);
            hit = (marks.Load((bit1 >> 5u) << 2u) >> (bit1 & 31u)) & 1u;
        }
        active = (hit != 0u);
    }

    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu)
            ^ uint4(tid, tid * 3u + 1u, tid * 7u + 2u, tid * 13u + 3u);

    if (active) {
        uint lo = piece * 65536u;
        uint hi = min(lo + 65536u, aligned_len);
        [loop]
        for (uint at = lo + tid * 16u; at < hi; at += 1024u) {
            uint4 v = src.Load4(at);
            h = (h ^ v) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
    }

    s_hash[tid] = h;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 32u; stride > 1u; stride >>= 1u) {
        if (active && tid < stride) {
            s_hash[tid] = mix_pair(s_hash[tid], s_hash[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        if (!active) {
            flags.Store(out_offset + gid.x * 4u, 0u);
        } else {
            uint4 final_h = mix_pair(s_hash[0], s_hash[1]);
            if (mode != 0u) {
                uint4 prev = hashes.Load4(piece * 16u);
                uint changed = any(final_h != prev) ? 1u : 0u;
                flags.Store(out_offset + gid.x * 4u, changed);
                if (changed != 0u) {
                    hashes.Store4(piece * 16u, final_h);
                }
            } else {
                hashes.Store4(piece * 16u, final_h);
            }
        }
    }
}
