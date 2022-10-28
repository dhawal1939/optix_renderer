#pragma once

#define MAX_LTC_LIGHTS 20
#define PI 3.141592653589793238


#define LOG(message)                                            \
    std::cout << OWL_TERMINAL_BLUE;                             \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
    std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;