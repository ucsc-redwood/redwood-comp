#pragma once

#include "cu_buffer.cuh"
#include <iterator>

namespace cuda {

template <typename T> class TypedBuffer final : public Buffer {
public:
  // Add iterator type aliases
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using iterator = T *;
  using const_iterator = const T *;

  TypedBuffer() = delete;

  explicit TypedBuffer(const size_t count)
      : Buffer(count * sizeof(T)), mapped_typed_data_(data()), count_(count) {}

  // Prevent copying, allow moving
  TypedBuffer(const TypedBuffer &) = delete;
  TypedBuffer &operator=(const TypedBuffer &) = delete;
  TypedBuffer(TypedBuffer &&) noexcept = default;
  TypedBuffer &operator=(TypedBuffer &&) noexcept = default;

  // Iterator support
  [[nodiscard]] iterator begin() { return mapped_typed_data_; }
  [[nodiscard]] const_iterator begin() const { return mapped_typed_data_; }
  [[nodiscard]] const_iterator cbegin() const { return mapped_typed_data_; }

  [[nodiscard]] iterator end() { return mapped_typed_data_ + count_; }
  [[nodiscard]] const_iterator end() const {
    return mapped_typed_data_ + count_;
  }
  [[nodiscard]] const_iterator cend() const {
    return mapped_typed_data_ + count_;
  }

  // Size information
  [[nodiscard]] size_type size() const { return count_; }
  [[nodiscard]] bool empty() const { return count_ == 0; }

  // Existing methods
  [[nodiscard]] T *data() { return reinterpret_cast<T *>(mapped_data_); }
  [[nodiscard]] const T *data() const {
    return reinterpret_cast<const T *>(mapped_data_);
  }

  [[nodiscard]] T &at(const size_t index) { return mapped_typed_data_[index]; }
  [[nodiscard]] const T &at(const size_t index) const {
    return mapped_typed_data_[index];
  }

private:
  T *mapped_typed_data_;
  const size_t count_;
};

} // namespace cuda
