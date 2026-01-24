cmake_minimum_required(VERSION 3.10)
project(MoogLadderExample VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MoogLadderExample
    example/main.cpp
    src/AudioDevice.cpp
    third_party/rtaudio/RtAudio.cpp
)

target_include_directories(MoogLadderExample PRIVATE
    src
    third_party
)

# Platform-specific RtAudio configuration
if(WIN32)
    target_compile_definitions(MoogLadderExample PRIVATE __WINDOWS_DS__)
    target_link_libraries(MoogLadderExample PRIVATE dsound ole32 winmm)
elseif(APPLE)
    target_compile_definitions(MoogLadderExample PRIVATE __MACOSX_CORE__)
    find_library(COREAUDIO_LIB CoreAudio REQUIRED)
    find_library(COREFOUNDATION_LIB CoreFoundation REQUIRED)
    target_link_libraries(MoogLadderExample PRIVATE ${COREAUDIO_LIB} ${COREFOUNDATION_LIB})
else()
    # Linux - use ALSA
    target_compile_definitions(MoogLadderExample PRIVATE __LINUX_ALSA__)
    find_package(ALSA REQUIRED)
    target_link_libraries(MoogLadderExample PRIVATE ALSA::ALSA pthread)
endif()
