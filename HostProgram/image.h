#pragma once

#include "common.h"
#include <VLR/vlrcpp.h>

vlr::Image2DRef loadImage2D(const vlr::ContextRef &context, const std::string &filepath, const std::string &spectrumType, const std::string &colorSpace);

void writePNG(const std::filesystem::path &filePath, uint32_t width, uint32_t height, const uint32_t* data);
void writeEXR(const std::filesystem::path &filePath, uint32_t width, uint32_t height, const float* data);