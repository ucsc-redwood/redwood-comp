#include "cu_kernels.cuh"

namespace cuda {

namespace kernels {

namespace dense {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

// start, end = 0, weight_matrix.rows;
__global__ void conv2d(const float* input_data,
                       int image_input_channels,
                       int input_height,
                       int input_width,
                       const CSRMatrix& weight_matrix,
                       const float* bias_data,
                       int bias_size,
                       int kernel_size,
                       int stride,
                       int padding,
                       bool relu,
                       float* output_data){

                       }

// ----------------------------------------------------------------------------
// Max Pooling 2D (Sparse)
// ----------------------------------------------------------------------------

__global__ void maxpool2d(const float* input_data,
                          int input_channels,
                          int input_height,
                          int input_width,
                          int pool_size,
                          int stride,
                          float* output_data){

                          }

// ----------------------------------------------------------------------------
// Linear Layer (Sparse)
// ----------------------------------------------------------------------------

__global__ void linear(const float* input_data,
                       const CSRMatrix& weight_matrix,
                       const float* bias_data,
                       float* output_data){
                        
                       }

}  // namespace dense
}  // namespace kernels
}  // namespace cuda