#ifndef LENET_INFERENCE_TENSOR_H
#define LENET_INFERENCE_TENSOR_H

#include <cstdlib>
#include <vector>
#include <array>

#ifndef aligned_alloc
// This may happen on OS X. Use POSIX's memalign to fake one.
inline void *aligned_alloc(size_t alignment, size_t size) {
  void *p;
  posix_memalign(&p, alignment, size);
  return p;
}
#endif

/**
 * @tparam N_DIM Number of dimension of the tensor.
 */
template<typename DTYPE, int N_DIM>
class Tensor {
 public:
  static constexpr size_t kAlignment = 128;

  inline int numElements() const {
    int num_elements = N_DIM ? dims_[0] : 0;
    for (int i = 1; i < N_DIM; ++i) {
      num_elements *= dims_[i];
    }
    return num_elements;
  }

  inline Tensor() {
    data_ = new DTYPE[0];
  }

  inline explicit Tensor(const std::array<int, N_DIM>& dims) : dims_(dims) {
    data_ = aligned_alloc(kAlignment, numElements() * sizeof(DTYPE));
  }

  inline void resize(const std::array<int, N_DIM>& dims) {
    bool size_changed = false;
    for (int i = 0; i < N_DIM; ++i) {
      if (dims[i] != dims_[i]) {
        size_changed = true;
        break;
      }
    }
    if (!size_changed) {
      return;
    }
    dims_ = dims;
    delete[] data_;
    data_ = aligned_alloc(kAlignment, numElements() * sizeof(DTYPE));
  }

  inline ~Tensor() {
    delete[] data_;
  }

  inline DTYPE* data() {
    return static_cast<DTYPE *>(data_);
  }
 protected:
  std::array<int, N_DIM> dims_;
  void* data_ = nullptr;
};

#endif //LENET_INFERENCE_TENSOR_H
