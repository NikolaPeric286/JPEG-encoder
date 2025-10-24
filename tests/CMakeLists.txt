if(BUILD_TESTING)
  add_executable(unit_tests tga_tests.cpp)
  target_link_libraries(unit_tests PRIVATE backend GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(unit_tests)
endif()