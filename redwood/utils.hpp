#pragma once

#include <spdlog/spdlog.h>

#include <source_location>

#define SPD_TRACE_FUNC \
  spdlog::trace(std::source_location::current().function_name());
