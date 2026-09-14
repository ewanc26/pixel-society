cmake_minimum_required(VERSION 3.20)
project(PixelSociety VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(PIXEL_SOCIETY_GUI "Build the SDL2 pixel observatory" ON)
option(PIXEL_SOCIETY_SANITIZE "Enable address and undefined behavior sanitizers" OFF)

add_library(society_core src/simulation.cpp src/neural.cpp src/parallel.cpp)
target_include_directories(society_core PUBLIC src)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(society_core PUBLIC -Wall -Wextra -Wpedantic)
    if(PIXEL_SOCIETY_SANITIZE)
        # AddressSanitizer and UBSan remain fully enabled at -O1. Debug's
        # default -O0 makes the sustained wide-core neural-world coverage too
        # slow to complete in CI, while modest optimization preserves useful
        # source-level diagnostics and keeps the sanitizer suite practical.
        target_compile_options(society_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -O1)
        target_link_options(society_core PUBLIC -fsanitize=address,undefined)
    endif()
endif()

add_executable(pixel-society src/main.cpp)
target_link_libraries(pixel-society PRIVATE society_core)
if(PIXEL_SOCIETY_GUI)
    find_package(SDL2 2.0.18 REQUIRED CONFIG)
    target_sources(pixel-society PRIVATE src/ui.cpp)
    target_compile_definitions(pixel-society PRIVATE PIXEL_SOCIETY_GUI=1)
    target_link_libraries(pixel-society PRIVATE SDL2::SDL2)
endif()
install(TARGETS pixel-society RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
    add_executable(society-tests tests/core_tests.cpp)
    target_link_libraries(society-tests PRIVATE society_core)
    if(PIXEL_SOCIETY_SANITIZE)
        # The release suite exercises a long multi-generation run. Sanitizers
        # exercise the same paths with a shorter sustained run so CI keeps its
        # memory-safety signal without timing out on instrumented code.
        target_compile_definitions(society-tests PRIVATE PIXEL_SOCIETY_SANITIZE_BUILD=1)
    endif()
    add_test(NAME core COMMAND society-tests)
    set_tests_properties(core PROPERTIES TIMEOUT 180)
    add_test(NAME help COMMAND pixel-society --help)
    add_test(NAME headless COMMAND pixel-society --headless --seed 42 --ticks 600 --events ${CMAKE_CURRENT_BINARY_DIR}/events.jsonl)
    add_test(NAME invalid_cli COMMAND pixel-society --founders -1 --headless)
    set_tests_properties(invalid_cli PROPERTIES WILL_FAIL TRUE)
    if(PIXEL_SOCIETY_GUI)
        add_test(NAME interface COMMAND pixel-society --smoke-test --screenshot ${CMAKE_CURRENT_BINARY_DIR}/smoke.bmp)
        set_tests_properties(interface PROPERTIES ENVIRONMENT "SDL_VIDEODRIVER=dummy" TIMEOUT 40)
    endif()
endif()
