// Block Jacobi for symmetric eigendecomposition.
//
// Standard cyclic Jacobi at n=256 hits the unified-memory bandwidth
// wall: every (p, q) rotation touches 2 rows + 2 cols + 2 V-cols
// across the WHOLE matrix, with no data reuse between rotations. With
// n × n × 4 = 256 KB > L2 cache, that is bandwidth-bound.
//
// Block Jacobi (Demmel, *Applied Numerical Linear Algebra*, ch. 5):
// partition the matrix into b × b sub-blocks. Process pairs of block
// columns; for each block-pair (i, j) form the 2b × 2b sub-matrix
// Bij = [A_ii A_ij; A_ji A_jj], compute its eigendecomposition Bij =
// Q D Qᵀ in *threadgroup* memory (small enough to stay cache-hot),
// then propagate Q to the rest of A and to the accumulated V.
//
// Block size b is fixed at 32 for n divisible by 32 (so n = 64, 128,
// 256, …). 2b × 2b = 64 × 64 = 16 KB; together with a 16 KB scratch
// for sub-Q this comfortably fits in Apple Silicon's 32 KB
// threadgroup memory limit.

#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 32u
#define SUB_DIM 64u   // 2 * BLOCK_SIZE
#define SUB_NN 4096u  // SUB_DIM * SUB_DIM
#define SUB_NPAIRS 32u  // SUB_DIM / 2
#define SUB_NROUNDS 63u // SUB_DIM - 1

inline void givens(float a_pp, float a_qq, float a_pq, thread float& c, thread float& s) {
    if (fabs(a_pq) < FLT_MIN) { c = 1.0f; s = 0.0f; return; }
    float theta = (a_qq - a_pp) / (2.0f * a_pq);
    float t = sign(theta) / (fabs(theta) + sqrt(theta * theta + 1.0f));
    c = 1.0f / sqrt(1.0f + t * t);
    s = t * c;
}

// Run cyclic Jacobi to convergence on a SUB_DIM×SUB_DIM matrix held in
// threadgroup memory. Output: sub_A (approximately diagonal), sub_Q
// (eigenvectors as columns).
inline void inner_jacobi_sweep(
    threadgroup float* sub_A,
    threadgroup float* sub_Q,
    constant int* sub_schedule,   // (SUB_NROUNDS, SUB_NPAIRS, 2) flat
    threadgroup float* sub_cs,    // (SUB_NPAIRS * 2)
    threadgroup float* off_sum,
    float tol_abs,
    uint max_sweeps,
    uint tid,
    uint tg_size)
{
    for (uint sw = 0; sw < max_sweeps; sw++) {
        for (uint r = 0; r < SUB_NROUNDS; r++) {
            // (c, s) per pair.
            if (tid < SUB_NPAIRS) {
                int p = sub_schedule[r * SUB_NPAIRS * 2 + tid * 2 + 0];
                int q = sub_schedule[r * SUB_NPAIRS * 2 + tid * 2 + 1];
                float c, s;
                if (p < 0) { c = 1.0f; s = 0.0f; }
                else { givens(sub_A[p * SUB_DIM + p], sub_A[q * SUB_DIM + q], sub_A[p * SUB_DIM + q], c, s); }
                sub_cs[tid * 2 + 0] = c;
                sub_cs[tid * 2 + 1] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Row updates on sub_A.
            uint total = SUB_NPAIRS * SUB_DIM;
            for (uint w = tid; w < total; w += tg_size) {
                uint i = w / SUB_DIM;
                uint k = w % SUB_DIM;
                int p = sub_schedule[r * SUB_NPAIRS * 2 + i * 2 + 0];
                if (p < 0) continue;
                int q = sub_schedule[r * SUB_NPAIRS * 2 + i * 2 + 1];
                float c = sub_cs[i * 2 + 0];
                float s = sub_cs[i * 2 + 1];
                float a_pk = sub_A[p * SUB_DIM + k], a_qk = sub_A[q * SUB_DIM + k];
                sub_A[p * SUB_DIM + k] = c * a_pk - s * a_qk;
                sub_A[q * SUB_DIM + k] = s * a_pk + c * a_qk;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Column updates on sub_A and sub_Q (one-sided V update).
            for (uint w = tid; w < total; w += tg_size) {
                uint i = w / SUB_DIM;
                uint k = w % SUB_DIM;
                int p = sub_schedule[r * SUB_NPAIRS * 2 + i * 2 + 0];
                if (p < 0) continue;
                int q = sub_schedule[r * SUB_NPAIRS * 2 + i * 2 + 1];
                float c = sub_cs[i * 2 + 0];
                float s = sub_cs[i * 2 + 1];
                float a_kp = sub_A[k * SUB_DIM + p], a_kq = sub_A[k * SUB_DIM + q];
                sub_A[k * SUB_DIM + p] = c * a_kp - s * a_kq;
                sub_A[k * SUB_DIM + q] = s * a_kp + c * a_kq;
                float q_kp = sub_Q[k * SUB_DIM + p], q_kq = sub_Q[k * SUB_DIM + q];
                sub_Q[k * SUB_DIM + p] = c * q_kp - s * q_kq;
                sub_Q[k * SUB_DIM + q] = s * q_kp + c * q_kq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Convergence: sum of squares of strictly upper-triangular off-diags.
        if (tid == 0) *off_sum = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float local = 0.0f;
        for (uint w = tid; w < SUB_NN; w += tg_size) {
            uint i = w / SUB_DIM, j = w % SUB_DIM;
            if (i < j) { float a = sub_A[w]; local += a * a; }
        }
        for (uint t = 0; t < tg_size; t++) {
            if (tid == t) *off_sum += local;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (*off_sum < tol_abs * tol_abs) break;
    }
}

kernel void block_jacobi_eigh_kernel(
    device float* batched_A          [[buffer(0)]],   // (B, n, n) input/scratch
    device float* batched_V          [[buffer(1)]],   // (B, n, n) output eigvecs
    constant uint& n                 [[buffer(2)]],
    constant uint& num_blocks        [[buffer(3)]],   // n / BLOCK_SIZE
    constant uint& max_block_sweeps  [[buffer(4)]],
    constant int*  block_schedule    [[buffer(5)]],   // (block_n_rounds, block_n_pairs, 2)
    constant uint& block_n_rounds    [[buffer(6)]],
    constant uint& block_n_pairs     [[buffer(7)]],
    constant int*  sub_schedule      [[buffer(8)]],   // tournament inside the 2b sub-matrix
    constant float& tol_abs_inner    [[buffer(9)]],
    constant uint& max_inner_sweeps  [[buffer(10)]],
    threadgroup float* shared_mem    [[threadgroup(0)]],   // sub_A + sub_Q + sub_cs + off_sum
    uint tg_id                       [[threadgroup_position_in_grid]],
    uint tid                         [[thread_position_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]])
{
    device float* A = batched_A + tg_id * n * n;
    device float* V = batched_V + tg_id * n * n;

    // Threadgroup memory layout:
    //   sub_A [SUB_NN]
    //   sub_Q [SUB_NN]
    //   sub_cs[SUB_NPAIRS * 2]
    //   off_sum[1]
    threadgroup float* sub_A = shared_mem;
    threadgroup float* sub_Q = sub_A + SUB_NN;
    threadgroup float* sub_cs = sub_Q + SUB_NN;
    threadgroup float* off_sum = sub_cs + SUB_NPAIRS * 2;

    // Initialise V = I.
    for (uint w = tid; w < n * n; w += tg_size) {
        uint i = w / n, j = w % n;
        V[w] = (i == j) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint bs = 0; bs < max_block_sweeps; bs++) {

        for (uint r = 0; r < block_n_rounds; r++) {
            for (uint pp = 0; pp < block_n_pairs; pp++) {
                int bi = block_schedule[r * block_n_pairs * 2 + pp * 2 + 0];
                int bj = block_schedule[r * block_n_pairs * 2 + pp * 2 + 1];
                if (bi < 0) continue;

                uint i_off = (uint)bi * BLOCK_SIZE;
                uint j_off = (uint)bj * BLOCK_SIZE;

                // 1) Load Bij = [A_ii A_ij; A_ji A_jj] into sub_A.
                //    The 2b×2b matrix indexed by (rr, cc) where:
                //      rr in [0, b)  ↔  global row  i_off + rr
                //      rr in [b, 2b) ↔  global row  j_off + (rr - b)
                //      similarly for cc
                for (uint w = tid; w < SUB_NN; w += tg_size) {
                    uint rr = w / SUB_DIM;
                    uint cc = w % SUB_DIM;
                    uint gr = (rr < BLOCK_SIZE) ? (i_off + rr) : (j_off + rr - BLOCK_SIZE);
                    uint gc = (cc < BLOCK_SIZE) ? (i_off + cc) : (j_off + cc - BLOCK_SIZE);
                    sub_A[w] = A[gr * n + gc];
                    sub_Q[w] = (rr == cc) ? 1.0f : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // 2) Inner Jacobi: diagonalise sub_A, accumulate sub_Q.
                inner_jacobi_sweep(sub_A, sub_Q, sub_schedule, sub_cs, off_sum,
                                   tol_abs_inner, max_inner_sweeps, tid, tg_size);

                // 3) Apply sub_Qᵀ from the LEFT to A's row blocks i and j.
                //    Equivalently: the 2b active rows form a 2b × n submatrix
                //    R = [A_i_rows; A_j_rows]. We compute R := sub_Qᵀ R.
                //    Stream column by column: at each col `cc`, read 2b values
                //    (one from each active row), apply sub_Qᵀ in registers,
                //    write back.
                //
                //    Each thread handles a chunk of columns.
                for (uint cc = tid; cc < n; cc += tg_size) {
                    // Read 2b values from column `cc`.
                    float r_vals[SUB_DIM];
                    for (uint k = 0; k < SUB_DIM; k++) {
                        uint gr = (k < BLOCK_SIZE) ? (i_off + k) : (j_off + k - BLOCK_SIZE);
                        r_vals[k] = A[gr * n + cc];
                    }
                    // Apply sub_Qᵀ from the left: r_new[i] = sum_k sub_Q[k, i] * r_vals[k]
                    //   (sub_Qᵀ[i, k] = sub_Q[k, i])
                    float r_new[SUB_DIM];
                    for (uint i = 0; i < SUB_DIM; i++) {
                        float acc = 0.0f;
                        for (uint k = 0; k < SUB_DIM; k++) {
                            acc += sub_Q[k * SUB_DIM + i] * r_vals[k];
                        }
                        r_new[i] = acc;
                    }
                    for (uint k = 0; k < SUB_DIM; k++) {
                        uint gr = (k < BLOCK_SIZE) ? (i_off + k) : (j_off + k - BLOCK_SIZE);
                        A[gr * n + cc] = r_new[k];
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);

                // 4) Apply sub_Q from the RIGHT to A's column blocks i and j.
                //    The 2b active columns form an n × 2b submatrix
                //    C[r, c]. Compute C := C sub_Q.
                for (uint rr = tid; rr < n; rr += tg_size) {
                    float c_vals[SUB_DIM];
                    for (uint k = 0; k < SUB_DIM; k++) {
                        uint gc = (k < BLOCK_SIZE) ? (i_off + k) : (j_off + k - BLOCK_SIZE);
                        c_vals[k] = A[rr * n + gc];
                    }
                    float c_new[SUB_DIM];
                    for (uint c = 0; c < SUB_DIM; c++) {
                        float acc = 0.0f;
                        for (uint k = 0; k < SUB_DIM; k++) {
                            acc += c_vals[k] * sub_Q[k * SUB_DIM + c];
                        }
                        c_new[c] = acc;
                    }
                    for (uint k = 0; k < SUB_DIM; k++) {
                        uint gc = (k < BLOCK_SIZE) ? (i_off + k) : (j_off + k - BLOCK_SIZE);
                        A[rr * n + gc] = c_new[k];
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);

                // 5) Update accumulated V columns: V[:, block_i ∪ block_j] @= sub_Q.
                for (uint rr = tid; rr < n; rr += tg_size) {
                    float v_vals[SUB_DIM];
                    for (uint k = 0; k < SUB_DIM; k++) {
                        uint gc = (k < BLOCK_SIZE) ? (i_off + k) : (j_off + k - BLOCK_SIZE);
                        v_vals[k] = V[rr * n + gc];
                    }
                    float v_new[SUB_DIM];
                    for (uint c = 0; c < SUB_DIM; c++) {
                        float acc = 0.0f;
                        for (uint k = 0; k < SUB_DIM; k++) {
                            acc += v_vals[k] * sub_Q[k * SUB_DIM + c];
                        }
                        v_new[c] = acc;
                    }
                    for (uint k = 0; k < SUB_DIM; k++) {
                        uint gc = (k < BLOCK_SIZE) ? (i_off + k) : (j_off + k - BLOCK_SIZE);
                        V[rr * n + gc] = v_new[k];
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
    }
}
