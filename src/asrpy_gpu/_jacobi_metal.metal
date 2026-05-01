// Batched symmetric eigendecomposition via parallel Jacobi rotations.
//
// One threadgroup processes one matrix in the batch. Threads within a
// threadgroup cooperate to apply the n/2 non-overlapping Givens rotations
// of each round of the parallel cyclic Jacobi method (Brent-Luk-style
// tournament schedule passed in via the `schedule` buffer).
//
// All matrix data lives in device memory (works for any n; threadgroup
// memory is too small for n > ~64 on Apple Silicon). The cost is moved
// from launch overhead to global memory bandwidth, which on Apple
// Silicon's unified memory is plenty for our problem sizes.

#include <metal_stdlib>
using namespace metal;

// Compute (c, s) of the Givens rotation that zeros A[p,q] of a 2x2
// symmetric submatrix [[a_pp, a_pq], [a_pq, a_qq]].
inline void givens(float a_pp, float a_qq, float a_pq, thread float& c, thread float& s) {
    if (fabs(a_pq) < FLT_MIN) {
        c = 1.0f;
        s = 0.0f;
        return;
    }
    float theta = (a_qq - a_pp) / (2.0f * a_pq);
    float t = sign(theta) / (fabs(theta) + sqrt(theta * theta + 1.0f));
    c = 1.0f / sqrt(1.0f + t * t);
    s = t * c;
}

kernel void jacobi_eigh_kernel(
    device float* batched_A          [[buffer(0)]],   // (B, n, n) — symmetric input, overwritten
    device float* batched_V          [[buffer(1)]],   // (B, n, n) — output eigenvectors
    constant uint& n                 [[buffer(2)]],
    constant uint& max_sweeps        [[buffer(3)]],
    constant int*  schedule          [[buffer(4)]],   // (n_rounds, n_pairs, 2) flat
    constant uint& n_rounds          [[buffer(5)]],
    constant uint& n_pairs           [[buffer(6)]],
    constant float& tol_abs          [[buffer(7)]],   // off-diag Frobenius threshold
    threadgroup float* shared_mem    [[threadgroup(0)]],  // (n_pairs*2 + 1) floats
    uint  tg_id                      [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_position_in_threadgroup]],
    uint  tg_size                    [[threads_per_threadgroup]])
{
    // Pointers to this matrix' (n*n) memory.
    device float* A = batched_A + tg_id * n * n;
    device float* V = batched_V + tg_id * n * n;

    // Threadgroup memory layout: first n_pairs*2 floats hold (c, s) per
    // pair (used inside the round); next slot is the partial-sum
    // accumulator for the convergence check.
    threadgroup float* shared_cs = shared_mem;
    threadgroup float* off_sum_slot = shared_mem + n_pairs * 2;

    // Initialize V = I (each thread strides over elements).
    for (uint k = tid; k < n * n; k += tg_size) {
        uint i = k / n;
        uint j = k % n;
        V[k] = (i == j) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Outer loop: sweeps.
    for (uint sw = 0; sw < max_sweeps; sw++) {

        // Inner loop: tournament rounds within the sweep.
        for (uint r = 0; r < n_rounds; r++) {

            // Step 1: each thread that owns a pair computes its (c, s).
            if (tid < n_pairs) {
                int p = schedule[r * n_pairs * 2 + tid * 2];
                int q = schedule[r * n_pairs * 2 + tid * 2 + 1];

                float c, s;
                if (p < 0) {  // sentinel for "no pair this round" (odd-n bye)
                    c = 1.0f;
                    s = 0.0f;
                } else {
                    float a_pp = A[p * n + p];
                    float a_qq = A[q * n + q];
                    float a_pq = A[p * n + q];
                    givens(a_pp, a_qq, a_pq, c, s);
                }
                shared_cs[tid * 2 + 0] = c;
                shared_cs[tid * 2 + 1] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Steps 2-3: apply n_pairs rotations in parallel.
            // Work is indexed by (i, k): pair i, column k. We have
            // n_pairs * n total work items per row-side step. Distribute
            // across the full threadgroup using a strided loop.
            //
            // Pairs in a round are non-overlapping in indices, so all
            // updates are independent and safe to do without synchronisation
            // *within* the same step. We do still need a barrier between
            // row-side and column-side because the column-side reads what
            // row-side has just written.
            uint total_row_work = n_pairs * n;
            for (uint w = tid; w < total_row_work; w += tg_size) {
                uint i = w / n;
                uint k = w % n;
                int p = schedule[r * n_pairs * 2 + i * 2 + 0];
                if (p < 0) continue;
                int q = schedule[r * n_pairs * 2 + i * 2 + 1];
                float c = shared_cs[i * 2 + 0];
                float s = shared_cs[i * 2 + 1];
                float a_pk = A[p * n + k];
                float a_qk = A[q * n + k];
                A[p * n + k] = c * a_pk - s * a_qk;
                A[q * n + k] = s * a_pk + c * a_qk;
            }
            threadgroup_barrier(mem_flags::mem_device);

            // Column-side rotations on A and V. Same indexing scheme.
            for (uint w = tid; w < total_row_work; w += tg_size) {
                uint i = w / n;
                uint k = w % n;
                int p = schedule[r * n_pairs * 2 + i * 2 + 0];
                if (p < 0) continue;
                int q = schedule[r * n_pairs * 2 + i * 2 + 1];
                float c = shared_cs[i * 2 + 0];
                float s = shared_cs[i * 2 + 1];

                float a_kp = A[k * n + p];
                float a_kq = A[k * n + q];
                A[k * n + p] = c * a_kp - s * a_kq;
                A[k * n + q] = s * a_kp + c * a_kq;

                float v_kp = V[k * n + p];
                float v_kq = V[k * n + q];
                V[k * n + p] = c * v_kp - s * v_kq;
                V[k * n + q] = s * v_kp + c * v_kq;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }

        // End-of-sweep convergence check: sum of squares of the strictly
        // upper-triangular off-diagonal entries. If below tol_abs² we
        // break out early. Each thread accumulates its strided portion;
        // thread 0 reduces and stores in off_sum_slot.
        if (tid == 0) {
            *off_sum_slot = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint w = tid; w < n * n; w += tg_size) {
            uint i = w / n;
            uint j = w % n;
            if (i < j) {
                float a = A[w];
                local_sum += a * a;
            }
        }
        // Naive reduction via atomic add on a single shared float.
        // Metal does not allow atomics on shared float directly; use a
        // round-robin write pattern: each thread adds its partial in turn.
        for (uint t = 0; t < tg_size; t++) {
            if (tid == t) {
                *off_sum_slot += local_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (*off_sum_slot < tol_abs * tol_abs) {
            break;  // early exit for this matrix
        }
    }

    // After sweeps: A is (approximately) diagonal. The kernel returns V;
    // eigenvalues are read by the host from the diagonal of A. Sorting and
    // reordering happen Python-side (cheap).
}
