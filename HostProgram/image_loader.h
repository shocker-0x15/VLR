#pragma once

#include "common.h"
#include <VLR/vlrcpp.h>

vlr::Image2DRef loadImage2D(const vlr::ContextRef &context, const std::string &filepath, const std::string &spectrumType, const std::string &colorSpace);
