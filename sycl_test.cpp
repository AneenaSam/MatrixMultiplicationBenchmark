#include <sycl/sycl.hpp>
#include <iostream>
int main() {
    for (auto& p : sycl::platform::get_platforms()) {
        std::cout << "Platform: " << p.get_info<sycl::info::platform::name>() << std::endl;
        for (auto& d : p.get_devices()) {
            std::cout << "  Device: " << d.get_info<sycl::info::device::name>();
            std::cout << (d.is_gpu() ? " [GPU]" : " [CPU]") << std::endl;
        }
    }
    std::cout << "Total platforms: " << sycl::platform::get_platforms().size() << std::endl;
    return 0;
}
