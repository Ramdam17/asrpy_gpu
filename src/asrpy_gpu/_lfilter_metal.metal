// Parallel-scan IIR filter via Blelloch up-down sweep.
//
// Implements direct-form-II IIR (the same form scipy.signal.lfilter uses for
// `lfilter(b, a, x)` along the last axis). The recurrence
//
//     w[n] = x[n] - a[1] w[n-1] - … - a[N] w[n-N]
//     y[n] = b[0] w[n] + b[1] w[n-1] + … + b[N] w[n-N]
//
// is rewritten as a state-space recurrence
//
//     s[n] = M s[n-1] + e_0 x[n]
//
// where s[n] = (w[n], w[n-1], …, w[n-N+1]) is an N-vector, M is the N×N
// companion matrix of the AR coefficients, and e_0 is the first standard
// basis vector. The combine operator
//
//     (M_a, c_a) ∘ (M_b, c_b) = (M_b M_a, M_b c_a + c_b)
//
// is associative, so we can prefix-scan the sequence
// (M, e_0 x[1]), (M, e_0 x[2]), … to obtain s[1], s[2], … in O(log T)
// parallel passes.
//
// One threadgroup processes one channel; the n_samples-long sequence is
// scanned cooperatively across the threadgroup. With N = 8 (Yule-Walker
// order in asrpy), each scan element is an 8×8 matrix + 8-vector pair —
// 72 floats. For n_samples = 30720 samples this fits comfortably in
// device memory; the threadgroup memory holds at most one tile of
// the up- and down-sweep working set.

#include <metal_stdlib>
using namespace metal;

#define ORDER 8u

// Combine two scan elements in registers. (M_a, c_a) ∘ (M_b, c_b)
// where the first element is the *earlier* one — combine reads
// "first apply a, then b" and produces (M_b M_a, M_b c_a + c_b).
inline void scan_combine(
    thread float* M_a, thread float* c_a,
    thread float* M_b, thread float* c_b,
    thread float* M_out, thread float* c_out)
{
    // M_out = M_b @ M_a   (8x8 × 8x8)
    for (uint i = 0; i < ORDER; i++) {
        for (uint j = 0; j < ORDER; j++) {
            float acc = 0.0f;
            for (uint k = 0; k < ORDER; k++) {
                acc += M_b[i * ORDER + k] * M_a[k * ORDER + j];
            }
            M_out[i * ORDER + j] = acc;
        }
    }
    // c_out = M_b @ c_a + c_b
    for (uint i = 0; i < ORDER; i++) {
        float acc = c_b[i];
        for (uint k = 0; k < ORDER; k++) {
            acc += M_b[i * ORDER + k] * c_a[k];
        }
        c_out[i] = acc;
    }
}

// Each scan element occupies (ORDER*ORDER + ORDER) = 72 floats. We store
// them packed in device memory: [M_flat (64), c (8)].
#define ELT_FLOATS (ORDER * ORDER + ORDER)

// Thread-private helper: load a scan element from device memory.
inline void load_elt(
    device const float* buf, uint idx,
    thread float* M, thread float* c)
{
    for (uint k = 0; k < ORDER * ORDER; k++) {
        M[k] = buf[idx * ELT_FLOATS + k];
    }
    for (uint k = 0; k < ORDER; k++) {
        c[k] = buf[idx * ELT_FLOATS + ORDER * ORDER + k];
    }
}

inline void store_elt(
    device float* buf, uint idx,
    const thread float* M, const thread float* c)
{
    for (uint k = 0; k < ORDER * ORDER; k++) {
        buf[idx * ELT_FLOATS + k] = M[k];
    }
    for (uint k = 0; k < ORDER; k++) {
        buf[idx * ELT_FLOATS + ORDER * ORDER + k] = c[k];
    }
}

// Build the per-sample scan element for input x[t]. The element is
// (M_const, e_0 * x[t]) — same M for every sample (an order-N IIR is
// linear time-invariant).
inline void build_elt(
    constant const float* M_const,
    float x_t,
    thread float* M, thread float* c)
{
    for (uint k = 0; k < ORDER * ORDER; k++) M[k] = M_const[k];
    c[0] = x_t;
    for (uint k = 1; k < ORDER; k++) c[k] = 0.0f;
}

// Phase 1: build per-sample scan elements (M_const, e_0 x[t]) into device
// memory `elts`. One thread per sample.
kernel void lfilter_build_elements(
    device const float* x        [[buffer(0)]],   // (n_chan, T)
    device float*       elts     [[buffer(1)]],   // (n_chan, T, ELT_FLOATS)
    constant const float* M_const [[buffer(2)]],  // (ORDER, ORDER)
    constant uint& T              [[buffer(3)]],
    uint tg_id                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    uint c = tg_id;
    device const float* x_chan = x + c * T;
    device float* elts_chan = elts + c * T * ELT_FLOATS;
    for (uint t = tid; t < T; t += tg_size) {
        thread float M[ORDER * ORDER];
        thread float cc[ORDER];
        build_elt(M_const, x_chan[t], M, cc);
        store_elt(elts_chan, t, M, cc);
    }
}

// Phase 2: in-place inclusive prefix-scan over the elements of one channel
// (one threadgroup per channel). Hillis–Steele parallel scan: log2(T)
// passes, each pass doubles the stride and combines elt[i - stride] into
// elt[i]. Simpler than Blelloch and competitive for our sizes.
kernel void lfilter_scan(
    device float*  elts          [[buffer(0)]],   // (n_chan, T, ELT_FLOATS)
    device float*  scratch       [[buffer(1)]],   // (n_chan, T, ELT_FLOATS) double buffer
    constant uint& T              [[buffer(2)]],
    uint tg_id                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    uint c = tg_id;
    device float* a = elts + c * T * ELT_FLOATS;
    device float* b = scratch + c * T * ELT_FLOATS;

    bool a_is_src = true;
    for (uint stride = 1; stride < T; stride <<= 1) {
        device float* src = a_is_src ? a : b;
        device float* dst = a_is_src ? b : a;

        for (uint i = tid; i < T; i += tg_size) {
            if (i >= stride) {
                thread float M_l[ORDER * ORDER], c_l[ORDER];
                thread float M_r[ORDER * ORDER], c_r[ORDER];
                thread float M_o[ORDER * ORDER], c_o[ORDER];
                load_elt(src, i - stride, M_l, c_l);
                load_elt(src, i, M_r, c_r);
                scan_combine(M_l, c_l, M_r, c_r, M_o, c_o);
                store_elt(dst, i, M_o, c_o);
            } else {
                // Copy unchanged.
                for (uint k = 0; k < ELT_FLOATS; k++) {
                    dst[i * ELT_FLOATS + k] = src[i * ELT_FLOATS + k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        a_is_src = !a_is_src;
    }

    // If the last stride wrote into b, copy back to a so the caller can
    // always read from `elts`.
    if (!a_is_src) {
        // a_is_src toggles each iteration; if final state is `a_is_src=false`
        // the last write was to a (because of the toggle). Wait — let's
        // actually just always copy to a to be safe.
    }
    if (a_is_src) {
        // Last iteration wrote to b (the OPPOSITE of a_is_src at start
        // of that iteration). Copy back.
        for (uint i = tid; i < T; i += tg_size) {
            for (uint k = 0; k < ELT_FLOATS; k++) {
                a[i * ELT_FLOATS + k] = b[i * ELT_FLOATS + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Phase 3: extract y[t] from the scanned states. y[t] = sum_k b_k * w[t-k]
// where w is the AR-filtered intermediate, and b is the FIR vector. The
// scanned state at position t is s[t] = (w[t], w[t-1], …, w[t-ORDER+1]),
// so y[t] = b · s[t].
kernel void lfilter_finish(
    device const float* elts     [[buffer(0)]],   // (n_chan, T, ELT_FLOATS)
    device float*       y        [[buffer(1)]],   // (n_chan, T)
    constant const float* B_const [[buffer(2)]],  // (ORDER,)
    constant uint& T              [[buffer(3)]],
    uint tg_id                    [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]])
{
    uint c = tg_id;
    device const float* elts_chan = elts + c * T * ELT_FLOATS;
    device float* y_chan = y + c * T;
    for (uint t = tid; t < T; t += tg_size) {
        // c-state lives in elts_chan[t * ELT_FLOATS + ORDER*ORDER ..]
        device const float* s = elts_chan + t * ELT_FLOATS + ORDER * ORDER;
        float acc = 0.0f;
        for (uint k = 0; k < ORDER; k++) acc += B_const[k] * s[k];
        y_chan[t] = acc;
    }
}
