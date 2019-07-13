#pragma once

#include "common.h"
#include <VLR/VLRCpp.h>

VLRCpp::Image2DRef loadImage2D(const VLRCpp::ContextRef &context, const std::string &filepath, const std::string &spectrumType, const std::string &colorSpace);
