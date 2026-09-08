#include <cassert>
#include <cmath>
#include <iostream>
#include "../src/tensor/tensor.hpp"

using namespace kayenai::tensor;

void test_tensor_creation() {
    std::cout << "Testing tensor creation...";
    
    Tensor t1({2, 3}, DType::FP32);
    assert(t1.numel() == 6);
    assert(t1.shape()[0] == 2);
    assert(t1.shape()[1] == 3);
    assert(t1.dtype() == DType::FP32);
    
    Tensor t2 = zeros({2, 3});
    assert(t2.ptr<float>()[0] == 0.0f);
    
    Tensor t3 = ones({2, 3});
    assert(t3.ptr<float>()[0] == 1.0f);
    
    std::cout << " PASS\n";
}

void test_tensor_fill() {
    std::cout << "Testing tensor fill...";
    
    Tensor t({3, 3}, DType::FP32);
    t.fill(5.0f);
    
    float* ptr = t.ptr<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        assert(ptr[i] == 5.0f);
    }
    
    std::cout << " PASS\n";
}

void test_tensor_reshape() {
    std::cout << "Testing tensor reshape...";
    
    Tensor t = ones({2, 3});
    Tensor reshaped = t.reshape({6});
    
    assert(reshaped.shape()[0] == 6);
    assert(reshaped.numel() == 6);
    assert(reshaped.ptr<float>()[0] == 1.0f);
    
    // Invalid reshape should throw
    bool exception_caught = false;
    try {
        t.reshape({5});
    } catch (const std::runtime_error&) {
        exception_caught = true;
    }
    assert(exception_caught);
    
    std::cout << " PASS\n";
}

void test_tensor_transpose() {
    std::cout << "Testing tensor transpose...";
    
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t({2, 3}, DType::FP32, data);
    Tensor transposed = t.transpose();
    
    assert(transposed.shape()[0] == 3);
    assert(transposed.shape()[1] == 2);
    
    float* ptr = transposed.ptr<float>();
    assert(ptr[0] == 1.0f);
    assert(ptr[1] == 4.0f);
    assert(ptr[2] == 2.0f);
    assert(ptr[3] == 5.0f);
    
    std::cout << " PASS\n";
}

void test_tensor_clone() {
    std::cout << "Testing tensor clone...";
    
    Tensor t1 = ones({2, 3});
    Tensor t2 = t1.clone();
    
    assert(t2.shape() == t1.shape());
    assert(t2.ptr<float>()[0] == 1.0f);
    
    // Modify original
    t1.fill(0.0f);
    assert(t2.ptr<float>()[0] == 1.0f);
    
    std::cout << " PASS\n";
}

void test_tensor_add() {
    std::cout << "Testing element-wise add...";
    
    float data1[] = {1.0f, 2.0f, 3.0f};
    float data2[] = {4.0f, 5.0f, 6.0f};
    
    Tensor t1({3}, DType::FP32, data1);
    Tensor t2({3}, DType::FP32, data2);
    Tensor result = add(t1, t2);
    
    float* ptr = result.ptr<float>();
    assert(ptr[0] == 5.0f);
    assert(ptr[1] == 7.0f);
    assert(ptr[2] == 9.0f);
    
    std::cout << " PASS\n";
}

void test_tensor_mul() {
    std::cout << "Testing element-wise mul...";
    
    float data1[] = {2.0f, 3.0f, 4.0f};
    float data2[] = {2.0f, 2.0f, 2.0f};
    
    Tensor t1({3}, DType::FP32, data1);
    Tensor t2({3}, DType::FP32, data2);
    Tensor result = mul(t1, t2);
    
    float* ptr = result.ptr<float>();
    assert(ptr[0] == 4.0f);
    assert(ptr[1] == 6.0f);
    assert(ptr[2] == 8.0f);
    
    std::cout << " PASS\n";
}

void test_tensor_matmul() {
    std::cout << "Testing matrix multiplication...";
    
    // (2,3) @ (3,2) = (2,2)
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    Tensor a({2, 3}, DType::FP32, a_data);
    Tensor b({3, 2}, DType::FP32, b_data);
    Tensor result = matmul(a, b);
    
    assert(result.shape()[0] == 2);
    assert(result.shape()[1] == 2);
    
    float* ptr = result.ptr<float>();
    // [1, 2, 3] @ [7, 8; 9, 10; 11, 12] = [58, 64; 139, 154]
    assert(std::abs(ptr[0] - 58.0f) < 1e-5);
    assert(std::abs(ptr[1] - 64.0f) < 1e-5);
    assert(std::abs(ptr[2] - 139.0f) < 1e-5);
    assert(std::abs(ptr[3] - 154.0f) < 1e-5);
    
    std::cout << " PASS\n";
}

void test_tensor_reductions() {
    std::cout << "Testing tensor reductions...";
    
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, DType::FP32, data);
    
    assert(std::abs(sum(t) - 10.0f) < 1e-5);
    assert(std::abs(mean(t) - 2.5f) < 1e-5);
    assert(std::abs(max(t) - 4.0f) < 1e-5);
    assert(std::abs(min(t) - 1.0f) < 1e-5);
    
    std::cout << " PASS\n";
}

void test_tensor_strides() {
    std::cout << "Testing tensor strides...";
    
    Tensor t({2, 3, 4}, DType::FP32);
    const auto& strides = t.strides();
    
    assert(strides[0] == 12);  // 3*4
    assert(strides[1] == 4);   // 4
    assert(strides[2] == 1);   // 1
    
    std::cout << " PASS\n";
}

void test_tensor_randn() {
    std::cout << "Testing random normal generation...";
    
    Tensor t = randn({100}, 0.0f, 1.0f);
    float m = mean(t);
    
    // Mean should be close to 0 with high probability (N=100)
    assert(std::abs(m) < 0.3f);
    
    std::cout << " PASS\n";
}

int main() {
    std::cout << "Running Tensor Unit Tests\n";
    std::cout << "==========================\n\n";
    
    try {
        test_tensor_creation();
        test_tensor_fill();
        test_tensor_reshape();
        test_tensor_transpose();
        test_tensor_clone();
        test_tensor_add();
        test_tensor_mul();
        test_tensor_matmul();
        test_tensor_reductions();
        test_tensor_strides();
        test_tensor_randn();
        
        std::cout << "\nAll tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
