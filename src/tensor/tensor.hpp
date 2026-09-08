#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <numeric>

namespace kayenai {
namespace tensor {

enum class DType {
    FP32,  // 32-bit float
    FP16,  // 16-bit float (half precision)
    BF16,  // bfloat16
    INT32, // 32-bit int
    INT8,  // 8-bit int
};

constexpr size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::INT32: return 4;
        case DType::INT8: return 1;
    }
    return 0;
}

class Tensor {
public:
    // Constructors
    Tensor() = default;
    
    // Create uninitialized tensor with given shape and dtype
    Tensor(const std::vector<size_t>& shape, DType dtype = DType::FP32);
    
    // Create tensor from raw data
    Tensor(const std::vector<size_t>& shape, DType dtype, const void* data);
    
    // Destructor
    ~Tensor() = default;
    
    // Deleted copy constructor/assignment (move semantics preferred)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Move semantics
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    
    // Properties
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t numel() const { return numel_; }
    size_t nbytes() const { return numel_ * dtype_size(dtype_); }
    DType dtype() const { return dtype_; }
    
    // Dimensions
    size_t ndim() const { return shape_.size(); }
    size_t dim(size_t i) const { return shape_[i]; }
    
    // Data access
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }
    
    // Get typed pointer (caller must ensure dtype matches)
    template<typename T>
    T* ptr() { return static_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* ptr() const { return static_cast<const T*>(data_.get()); }
    
    // Reshape (returns new tensor with same data but different shape)
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    
    // Transpose (2D tensors)
    Tensor transpose() const;
    
    // Clone with new allocation
    Tensor clone() const;
    
    // Fill with constant
    void fill(float value);
    
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t numel_ = 0;
    DType dtype_ = DType::FP32;
    std::shared_ptr<void> data_;
    
    void compute_strides();
    void allocate();
};

// Tensor creation helpers
Tensor zeros(const std::vector<size_t>& shape, DType dtype = DType::FP32);
Tensor ones(const std::vector<size_t>& shape, DType dtype = DType::FP32);
Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float std = 1.0f);

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

// Reductions
float sum(const Tensor& t);
float mean(const Tensor& t);
float max(const Tensor& t);
float min(const Tensor& t);

} // namespace tensor
} // namespace kayenai
