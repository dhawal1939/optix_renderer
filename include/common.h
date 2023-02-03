#pragma once

#define MAX_LTC_LIGHTS 20
#define PI 3.141592653589793238f
#define TWO_PI 2*3.141592653589793238f
#define INV_TWO_PI  0.1591549f
#define VEC4f owl::common::vec4f
#define VEC3f owl::common::vec3f
#define VEC2f owl::common::vec2f
#define VEC4 owl::common::vec4f
#define VEC3 owl::common::vec3f
#define VEC2 owl::common::vec2f

#define VEC3i owl::common::vec3i
#define VEC2i owl::common::vec2i


#define LOG(message)                                            \
    std::cout << OWL_TERMINAL_BLUE;                             \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
    std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;