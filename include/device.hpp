#ifndef GENERALS_DEVICE_HPP
#define GENERALS_DEVICE_HPP

#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/cuda.h>

namespace generals {

inline c10::Device get_device() {
  if (torch::cuda::is_available())
    return {c10::DeviceType::CUDA};
  else
    return {c10::DeviceType::CPU};
}

} // namespace generals

#endif
