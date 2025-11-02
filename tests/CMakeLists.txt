if(BUILD_TESTING)
  file(GLOB source_files "*.cpp")

  add_executable(unit_tests ${source_files})
  target_link_libraries(unit_tests PRIVATE backend GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(unit_tests)
endif()