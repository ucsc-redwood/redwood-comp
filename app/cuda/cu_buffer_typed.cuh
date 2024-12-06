#pragma once

#include "cu_buffer.cuh"

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace cuda {

template <typename T>
class TypedBuffer final : public Buffer,
                          public std::enable_shared_from_this<TypedBuffer<T>> {
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
  [[nodiscard]] size_type size_bytes() const { return count_ * sizeof(T); }
  [[nodiscard]] bool empty() const { return count_ == 0; }

  // Data access
  [[nodiscard]] T *data() { return reinterpret_cast<T *>(mapped_data_); }
  [[nodiscard]] const T *data() const {
    return reinterpret_cast<const T *>(mapped_data_);
  }

  [[nodiscard]] T &at(const size_t index) { return mapped_typed_data_[index]; }
  [[nodiscard]] const T &at(const size_t index) const {
    return mapped_typed_data_[index];
  }

  // --------------------------------------------------------------------------
  // Helper functions to quickly fill the buffer as you construct it
  // --------------------------------------------------------------------------

  [[nodiscard]] auto fill(const T &value) -> std::shared_ptr<TypedBuffer<T>> {
    std::ranges::fill(*this, value);
    return this->shared_from_this();
  }

  [[nodiscard]] auto zeros() -> std::shared_ptr<TypedBuffer<T>> {
    if constexpr (std::is_trivially_constructible_v<T>) {
      std::memset(mapped_typed_data_, 0, size_bytes());
      return this->shared_from_this();
    } else {
      return fill(T{});
    }
  }

  [[nodiscard]] auto ones() -> std::shared_ptr<TypedBuffer<T>> {
    return fill(T{1});
  }

  template <typename RNG = std::mt19937>
  [[nodiscard]] auto
  random(const T min = T{}, const T max = std::numeric_limits<T>::max(),
         const typename RNG::result_type seed = std::random_device{}())
      -> std::shared_ptr<TypedBuffer<T>> {
    RNG gen(seed);

    if constexpr (std::floating_point<T>) {
      std::uniform_real_distribution<T> dis(min, max);
      std::ranges::generate(*this, [&]() { return dis(gen); });
    } else if constexpr (std::integral<T>) {
      std::uniform_int_distribution<T> dis(min, max);
      std::ranges::generate(*this, [&]() { return dis(gen); });
    } else {
      static_assert(std::floating_point<T> || std::integral<T>,
                    "random() only supports floating point or integral types");
    }
    return this->shared_from_this();
  }

  auto
  from_vector(const std::vector<T> &vec) -> std::shared_ptr<TypedBuffer<T>> {
    assert(vec.size() == count_);
    std::ranges::copy(vec, mapped_typed_data_);
    return this->shared_from_this();
  }

private:
  T *mapped_typed_data_;
  const size_t count_;
};

} // namespace cuda
