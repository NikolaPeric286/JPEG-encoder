file(GLOB source_files "*.cpp")
add_executable(jpeg_encoder_cli ${source_files})
target_link_libraries(jpeg_encoder_cli PRIVATE backend libraw::libraw)