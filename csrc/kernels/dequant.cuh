// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "util/cuda_utils.cuh"

namespace vptq::kernels {

template <typename scalar_t, int IDXBITS, int ResidualBits, int GROUPSIZE,
          bool Return_OUF_x_INF>
__global__ void DequantizeWithOutliers_PackIndice(
    scalar_t* out, const int32_t* q_indice, const int16_t* q_indice_outliers,
    const scalar_t* centroids, 
    const scalar_t* residual_centroids,
    const scalar_t* outliers_centroids, 
    const uint16_t* invert_perm,
    const scalar_t* weight_scale, const scalar_t* weight_bias, int out_features,
    int in_features, int outliers_infeatures, int OL_GroupSize,
    const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums, bool use_shared_memory) {
  // Shared memory allocation for centroid data (if used)
  extern __shared__ char s_data[];
  
  // Thread identifiers
  int bid = blockIdx.x;
  int tid = (bid * cuda::kBlockSize + threadIdx.x);
  int local_tid = threadIdx.x;
  int in_x = tid % in_features;
  int in_y = tid / in_features;
  using VecType = typename cuda::TypeVec2<scalar_t>::type;

  // Shared memory pointers (used only if shared memory is allocated)
  scalar_t* s_centroids = nullptr;
  scalar_t* s_residual_centroids = nullptr;
  scalar_t* s_outliers_centroids = nullptr;
  
  if (use_shared_memory) {
    // Use dynamic shared memory to store centroids, residual_centroids, and outliers_centroids
    // We'll only load the part of the centroids that this block needs
    const int max_centroids_size = (1 << IDXBITS);
    const int max_residual_size = ResidualBits > 0 ? (1 << ResidualBits) : 0;
    const int max_outliers_size = 256; // Assuming maximum outliers size
    
    // Shared memory partitioning
    s_centroids = (scalar_t*)s_data;
    s_residual_centroids = s_centroids + max_centroids_size * GROUPSIZE;
    s_outliers_centroids = s_residual_centroids + max_residual_size * GROUPSIZE;
    
    // Cooperatively load centroids into shared memory
    // Each thread loads multiple elements
    for (int i = local_tid; i < max_centroids_size * GROUPSIZE; i += blockDim.x) {
      int cent_idx = i / GROUPSIZE;
      int group_offset = i % GROUPSIZE;
      s_centroids[i] = centroids[cent_idx * GROUPSIZE + group_offset];
    }
    
    // Load residual centroids if needed
    if constexpr (ResidualBits > 0) {
      for (int i = local_tid; i < max_residual_size * GROUPSIZE; i += blockDim.x) {
        int res_idx = i / GROUPSIZE;
        int group_offset = i % GROUPSIZE;
        s_residual_centroids[i] = residual_centroids[res_idx * GROUPSIZE + group_offset];
      }
    }
    
    // Load outliers centroids (if needed)
    if (outliers_infeatures > 0) {
      for (int i = local_tid; i < max_outliers_size * OL_GroupSize; i += blockDim.x) {
        int ol_idx = i / OL_GroupSize;
        int group_offset = i % OL_GroupSize;
        if (ol_idx < max_outliers_size) {
          s_outliers_centroids[i] = outliers_centroids[ol_idx * OL_GroupSize + group_offset];
        }
      }
    }
    
    // Ensure all threads have loaded the shared memory
    __syncthreads();
  }

  uint16_t mapped_index_x = invert_perm ? invert_perm[in_x] : in_x;
  const scalar_t scale = weight_scale[in_x];
  const scalar_t bias = weight_bias[in_x];

  if (mapped_index_x < outliers_infeatures) {
    const int n_outlisers_groups_in_normalgroup = GROUPSIZE / OL_GroupSize;
    q_indice_outliers +=
        in_y * n_outlisers_groups_in_normalgroup * outliers_infeatures +
        mapped_index_x;
#pragma unroll(3)
    for (int i = 0; i < n_outlisers_groups_in_normalgroup; ++i) {
      if (in_y * n_outlisers_groups_in_normalgroup + i >=
          out_features / OL_GroupSize)
        return;
      const uint16_t outliers_ind = q_indice_outliers[(i)*outliers_infeatures];
      // Use shared memory for outliers_centroids if available
      const scalar_t* outliers_centroids_start = use_shared_memory ? 
          s_outliers_centroids + outliers_ind * OL_GroupSize :
          outliers_centroids + outliers_ind * OL_GroupSize;
      const int gi = in_y * GROUPSIZE + i * OL_GroupSize;
#pragma unroll(4)
      for (int j = 0; j < OL_GroupSize; ++j) {
        if ((gi + j) >= out_features) {
          return;
        }
        out[(gi + j) * in_features + in_x] =
            FMA(outliers_centroids_start[j], scale, bias);
      }
    }
    return;
  }

  const int inliers_infeatures_in_group =
      (in_features - outliers_infeatures) / group_nums;

  const int mapped_inliers_inx = (mapped_index_x - outliers_infeatures);
  const int code_books_id = mapped_inliers_inx / inliers_infeatures_in_group;
  const int mappped_inx_in_a_codebook =
      mapped_inliers_inx % inliers_infeatures_in_group;

  // Original global memory access pointers (we'll keep these for indexing)
  const int32_t* q_indice_global = q_indice;
  const scalar_t* centroids_global = centroids;
  const scalar_t* residual_centroids_global = residual_centroids;
  
  if (group_nums > 1) {  // has multiple codebooks
    q_indice_global += code_books_id * index_stride_0;
    if (!use_shared_memory) {
      centroids_global += code_books_id * centroids_stride_0;
      residual_centroids_global += code_books_id * centroids_stride_0;
    }
  }
  q_indice_global += in_y * index_stride_1;

  uint32_t merged_ind = cuda::iterator_packed_tensor<IDXBITS + ResidualBits>(
      (const uint32_t*)q_indice_global, mappped_inx_in_a_codebook);

  const uint16_t base_ind = merged_ind & ((1 << IDXBITS) - 1);
  VecType base[GROUPSIZE / 2];
  
  if (use_shared_memory) {
    // Use shared memory for centroids
    const scalar_t* centroids_start = s_centroids + base_ind * GROUPSIZE;
    
    // Load from shared memory
    for (int i = 0; i < GROUPSIZE / 2; i++) {
      base[i].x = centroids_start[i*2];
      base[i].y = centroids_start[i*2+1];
    }
  } else {
    // Use global memory for centroids
    const scalar_t* centroids_start = centroids_global + base_ind * GROUPSIZE;
    cuda::ldg_vec_x<GROUPSIZE>((base), (const uint32_t*)(centroids_start));
  }

  if constexpr (ResidualBits > 0) {
    VecType residual[GROUPSIZE / 2];
    merged_ind >>= IDXBITS;
    const uint16_t res_ind = merged_ind & ((1 << ResidualBits) - 1);
    
    if (use_shared_memory) {
      // Use shared memory for residual_centroids
      const scalar_t* residual_centroids_start =
          s_residual_centroids + res_ind * GROUPSIZE;
      
      // Load from shared memory
      for (int i = 0; i < GROUPSIZE / 2; i++) {
        residual[i].x = residual_centroids_start[i*2];
        residual[i].y = residual_centroids_start[i*2+1];
      }
    } else {
      // Use global memory for residual_centroids
      const scalar_t* residual_centroids_start =
          residual_centroids_global + res_ind * GROUPSIZE;
      cuda::ldg_vec_x<GROUPSIZE>((residual),
                               (const uint32_t*)(residual_centroids_start));
    }
    
#pragma unroll
    for (int i = 0; i < GROUPSIZE / 2; ++i) {
      base[i] = ADD2(base[i], residual[i]);
    }
  }

  VecType hres[GROUPSIZE / 2];
  VecType scale2 = VecType{scale, scale};
  VecType bias2 = VecType{bias, bias};
#pragma unroll
  for (int i = 0; i < GROUPSIZE / 2; ++i) {
    hres[i] = FMA2(base[i], scale2, bias2);
  }
  scalar_t* res = (scalar_t*)hres;
  const int group_step = in_y * GROUPSIZE;
  if constexpr (!Return_OUF_x_INF) {
    out += in_x * out_features + group_step;
  } else {
    out += (group_step)*in_features + in_x;
  }
#pragma unroll
  for (int i = 0; i < GROUPSIZE; ++i) {
    if ((group_step + i) < out_features) {
      if constexpr (Return_OUF_x_INF) {
        out[i * in_features] = res[i];
      } else {
        out[i] = res[i];
      }
    }
  }
}

}  // namespace vptq::kernels
