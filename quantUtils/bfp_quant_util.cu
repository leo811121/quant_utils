#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void bfp_quantize_kernel(const float *__restrict__ input, 
    int8_t *__restrict__ quantized, 
    int8_t * scales, 
    int batch, int height, int width) {

    int block_b = blockIdx.x;
    int block_h = blockIdx.y;
    int block_w = blockIdx.z;
    int thread_w = threadIdx.x;

    int row_start = block_b * height * width + block_h * width; // width = N * block_size
    int block_start = row_start + block_w * blockDim.x;
    int index = block_start + thread_w;

    extern __shared__ __half shared_data[];

    // move data to share memory
    if (index < batch * height * width) {
        shared_data[thread_w] = __float2half(input[index]);
    } else {
        shared_data[thread_w] = __float2half(0.0f);
    }
    __syncthreads();

    // store element into register
    __half input_reg = shared_data[thread_w];

    // find max element
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_w < stride) {
            shared_data[thread_w] = __hmax(__habs(shared_data[thread_w]), __habs(shared_data[thread_w + stride]));
        }
        __syncthreads();
    }

    uint8_t sign_mask = static_cast<uint8_t>((__half_as_ushort(input_reg) >> 15) << 7);
    int8_t max_exp_unbias = __heq(shared_data[0], __float2half(0.0f)) ? -15 : ((*(uint16_t*)&shared_data[0] >> 10) & 0x1F) - 15;
    int8_t exp_unbias = static_cast<uint8_t>(((__half_as_ushort(input_reg) >> 10) & 0x1F) - 15);
    int8_t r_shift = max_exp_unbias - exp_unbias;
    uint8_t quant_tensor = static_cast<uint8_t>((__half_as_ushort(input_reg) & 0x03FF) >> (4 + r_shift));
    quant_tensor |= sign_mask;

    if (exp_unbias != -15) {
        quant_tensor |= (0x40 >> r_shift);
    }
    
    if (index < batch * height * width) {
        quantized[index] = *(int8_t*)& quant_tensor;
    }

    if (thread_w == 0) {
        scales[block_start / blockDim.x] = max_exp_unbias;
    }
}

__global__ void bfp_dequantize_kernel(int8_t *quantized, float *dequantized, int8_t *scales, int batch, int height, int width) {
    int block_b = blockIdx.x;
    int block_h = blockIdx.y;
    int block_w = blockIdx.z;
    int thread_w = threadIdx.x;

    int row_start = block_b * height * width + block_h * width; // width = N * block_size
    int block_start = row_start + block_w * blockDim.x;
    int index = block_start + thread_w;

    int8_t scale = scales[block_start / blockDim.x];

    int l_shift = 0;
    uint8_t quant_tensor = quantized[index];

    uint8_t quantized_tmp = quantized[index] << 1;
    for (int i=0; i<8; i++) {
        uint8_t quantized_msb = quantized_tmp & 0x80;
        if (quantized_msb == 0x80) {
            l_shift = i;
            break;
        }else {
            l_shift = i;
            quantized_tmp = quantized_tmp << 1;
        }
    }
    __syncthreads();

    uint16_t sign_bit = (*(uint16_t*)&quant_tensor << 8) & 0x8000;
    int8_t _exp_bits = scale - l_shift + 15;
    uint16_t exp_bits = *(uint16_t*)&_exp_bits << 10;
    uint16_t mant_bits = (*(uint16_t*)&quant_tensor) << (4 + l_shift);
    mant_bits &= 0x03FF;
    uint16_t _dequant_tensor = sign_bit | exp_bits | mant_bits;
    float dequant_tensor = __half2float(*(__half*)&_dequant_tensor);
    dequantized[block_start + thread_w] = dequant_tensor;
}

std::tuple<torch::Tensor, torch::Tensor> bfp_quantize(const torch::Tensor input, const int block_size) {
    if (!input.device().is_cuda()) {
        throw std::runtime_error("Error: Tensor is not on GPU!");
    } else if  (input.dtype() != torch::kFloat) {
        throw std::runtime_error("Error: Tensor is not float32!");
    }

    const int batch = input.size(0);
    const int height = input.size(1);
    const int width = input.size(2);
    const int num_elements = batch * height * width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    dim3 gridDim(min(batch, 65535), min(height, 65535), width / block_size);
    dim3 blockDim(block_size);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    auto quantized = torch::full({batch, height, width}, 0, options);
    auto scales = torch::full({batch, height, num_blocks / (batch * height)}, 0, options);

    bfp_quantize_kernel<<<gridDim, blockDim>>>(input.data_ptr<float>(), 
                                               quantized.data_ptr<int8_t>(), 
                                               scales.data_ptr<int8_t>(), 
                                               batch, height, width);

    return std::make_tuple(quantized, scales);
}

torch::Tensor bfp_dequantize(const torch::Tensor quantized, const torch::Tensor scales) {
    if (!quantized.device().is_cuda() || !scales.device().is_cuda()) {
        throw std::runtime_error("Error: Tensor is not on GPU!");
    } else if (quantized.dtype() != torch::kInt8 || scales.dtype() != torch::kInt8) {
        throw std::runtime_error("Error: Tensor is not int8!");
    }

    const int batch = quantized.size(0);
    const int height = quantized.size(1);
    const int width = quantized.size(2);
    const int len_scales = scales.numel();
    const int num_elements = batch * height * width;
    int block_size = num_elements / len_scales;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    dim3 gridDim(min(batch, 65535), min(height, 65535), width / block_size);
    dim3 blockDim(block_size);

    auto dequantized = torch::full({batch, height, width}, 0, torch::TensorOptions().dtype(torch::kFloat).device(quantized.device()));

    bfp_dequantize_kernel<<<gridDim, blockDim>>>(quantized.data_ptr<int8_t>(),
                                                 dequantized.data_ptr<float>(), 
                                                 scales.data_ptr<int8_t>(), 
                                                 batch, height, width);

    return dequantized;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bfp_quantize", &bfp_quantize, "BFP quantize and return quantized tensor and scales");
    m.def("bfp_dequantize", &bfp_dequantize, "BFP dequantize and return dequantized tensor");
}
