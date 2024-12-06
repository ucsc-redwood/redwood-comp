// #include <memory_resource>

// template <typename T>
// class UsmVector {
//  public:
//   using allocator_type = std::pmr::polymorphic_allocator<T>;

//   explicit UsmVector(
//       std::pmr::memory_resource* mr = std::pmr::get_default_resource())
//       : alloc_(mr), data_(alloc_) {}

//   void push_back(const T& value) { data_.push_back(value); }

//   // Other vector-like operations...
//  private:
//   allocator_type alloc_;
//   std::pmr::vector<T> data_;
// };
