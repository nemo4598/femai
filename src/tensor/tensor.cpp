#include "tensor.hpp"
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

namespace kayenai {
namespace tensor {

// Tensor constructor
Tensor::Tensor(const std::vector<size_t>& shape, DType dtype)
    : shape_(shape), dtype_(dtype) {
    numel_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    compute_strides();
    allocate();
}

Tensor::Tensor(const std::vector<size_t>& shape, DType dtype, const void* data)
    : shape_(shape), dtype_(dtype) {
    numel_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    compute_strides();
    allocate();
    if (data) {
        std::memcpy(data_.get(), data, nbytes());
    }
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

void Tensor::allocate() {
    if (numel_ == 0) {
        data_ = nullptr;
        return;
    }
    
    size_t bytes = nbytes();
    void* ptr = std::malloc(bytes);
    if (!ptr) throw std::runtime_error("Memory allocation failed");
    
    data_ = std::shared_ptr<void>(ptr, [](void* p) { std::free(p); });
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (new_numel != numel_) {
        throw std::runtime_error("Cannot reshape: size mismatch");
    }
    
    Tensor result;
    result.shape_ = new_shape;
    result.dtype_ = dtype_;
    result.numel_ = new_numel;
    result.data_ = data_;
    result.compute_strides();
    
    return result;
}

Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::runtime_error("Transpose only supported for 2D tensors");
    }
    
    std::vector<size_t> new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape, dtype_);
    const float* src = ptr<float>();
    float* dst = result.ptr<float>();
    
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            dst[j * shape_[0] + i] = src[i * shape_[1] + j];
        }
    }
    
    return result;
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_, data_.get());
    return result;
}

void Tensor::fill(float value) {
    if (dtype_ != DType::FP32) {
        throw std::runtime_error("fill() currently only supports FP32");
    }
    float* ptr = this->ptr<float>();
    std::fill(ptr, ptr + numel_, value);
}

// Helpers
Tensor zeros(const std::vector<size_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::FP32) {
        std::fill(t.ptr<float>(), t.ptr<float>() + t.numel(), 0.0f);
    }
    return t;
}

Tensor ones(const std::vector<size_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::FP32) {
        std::fill(t.ptr<float>(), t.ptr<float>() + t.numel(), 1.0f);
    }
    return t;
}

Tensor randn(const std::vector<size_t>& shape, float mean, float std) {
    Tensor t(shape, DType::FP32);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(mean, std);
    
    float* ptr = t.ptr<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        ptr[i] = dist(gen);
    }
    return t;
}

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
        throw std::runtime_error("Shape or dtype mismatch in add");
    }
    
    Tensor result = a.clone();
    if (a.dtype() == DType::FP32) {
        float* res = result.ptr<float>();
        const float* b_ptr = b.ptr<float>();
        for (size_t i = 0; i < a.numel(); ++i) {
            res[i] += b_ptr[i];
        }
    }
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.dtype() != b.dtype()) {
        throw std::runtime_error("Shape or dtype mismatch in mul");
    }
    
    Tensor result = a.clone();
    if (a.dtype() == DType::FP32) {
        float* res = result.ptr<float>();
        const float* b_ptr = b.ptr<float>();
        for (size_t i = 0; i < a.numel(); ++i) {
            res[i] *= b_ptr[i];
        }
    }
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("matmul currently only supports 2D tensors");
    }
    if (a.dim(1) != b.dim(0)) {
        throw std::runtime_error("Dimension mismatch in matmul");
    }
    
    Tensor result({a.dim(0), b.dim(1)}, DType::FP32);
    result.fill(0.0f);
    
    const float* a_ptr = a.ptr<float>();
    const float* b_ptr = b.ptr<float>();
    float* res = result.ptr<float>();
    
    for (size_t i = 0; i < a.dim(0); ++i) {
        for (size_t k = 0; k < a.dim(1); ++k) {
            for (size_t j = 0; j < b.dim(1); ++j) {
                res[i * b.dim(1) + j] += a_ptr[i * a.dim(1) + k] * b_ptr[k * b.dim(1) + j];
            }
        }
    }
    
    return result;
}

// Reductions
float sum(const Tensor& t) {
    if (t.dtype() != DType::FP32) {
        throw std::runtime_error("sum() currently only supports FP32");
    }
    const float* ptr = t.ptr<float>();
    float result = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) {
        result += ptr[i];
    }
    return result;
}

float mean(const Tensor& t) {
    return sum(t) / static_cast<float>(t.numel());
}

float max(const Tensor& t) {
    if (t.dtype() != DType::FP32) {
        throw std::runtime_error("max() currently only supports FP32");
    }
    const float* ptr = t.ptr<float>();
    return *std::max_element(ptr, ptr + t.numel());
}

float min(const Tensor& t) {
    if (t.dtype() != DType::FP32) {
        throw std::runtime_error("min() currently only supports FP32");
    }
    const float* ptr = t.ptr<float>();
    return *std::min_element(ptr, ptr + t.numel());
}

} // namespace tensor
} // namespace kayenai
