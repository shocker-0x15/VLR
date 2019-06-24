#include "connectable.h"

namespace VLR {
    const char* ParameterFloat = "Float";
    const char* ParameterPoint3D = "Point3D";
    const char* ParameterVector3D = "Vector3D";
    const char* ParameterNormal3D = "Normal3D";
    const char* ParameterSpectrum = "Spectrum";
    const char* ParameterAlpha = "Alpha";
    const char* ParameterTextureCoordinates = "TextureCoordinates";

    const char* ParameterImage = "Image";
    const char* ParameterSurfaceMaterial = "SurfaceMaterial";

    const char* ParameterSpectrumType = "SpectrumType";
    const char* ParameterColorSpace = "ColorSpace";
    const char* ParameterBumpType = "BumpType";
    const char* ParameterTextureFilter = "TextureFilter";
    const char* ParameterTextureWrapMode = "TextureWrapMode";
    const char* ParameterCameraType = "CameraType";

    static bool s_enumTableInitialized = false;
    static const std::map<std::string, std::vector<std::pair<const char*, uint32_t>>> s_enumTables = {
        {
            ParameterSpectrumType, {
                {"Reflectance", (uint32_t)SpectrumType::Reflectance},
                {"Transmittance", (uint32_t)SpectrumType::Transmittance},
                {"LightSource", (uint32_t)SpectrumType::LightSource},
                {"NA", (uint32_t)SpectrumType::NA},
            }
        },
        {
            ParameterColorSpace, {
                {"Rec709(D65) sRGB Gamma", (uint32_t)ColorSpace::Rec709_D65_sRGBGamma},
                {"Rec709(D65)", (uint32_t)ColorSpace::Rec709_D65},
                {"XYZ", (uint32_t)ColorSpace::XYZ},
                {"xyY", (uint32_t)ColorSpace::xyY},
            }
        },
        {
            ParameterBumpType, {
                {"Normal Map (DirectX)", (uint32_t)BumpType::NormalMap_DirectX},
                {"Normal Map (OpenGL)", (uint32_t)BumpType::NormalMap_OpenGL},
                {"Height Map", (uint32_t)BumpType::HeightMap},
            }
        },
        {
            ParameterTextureFilter, {
                {"Nearest", (uint32_t)VLRTextureFilter_Nearest},
                {"Linear", (uint32_t)VLRTextureFilter_Linear},
                {"None", (uint32_t)VLRTextureFilter_None},
            }
        },
        {
            ParameterTextureWrapMode, {
                {"Repeat", (uint32_t)VLRTextureWrapMode_Repeat},
                {"Clamp to Edge", (uint32_t)VLRTextureWrapMode_ClampToEdge},
                {"Mirror", (uint32_t)VLRTextureWrapMode_Mirror},
                {"Clamp to Border", (uint32_t)VLRTextureWrapMode_ClampToBorder},
            }
        },
        {
            ParameterCameraType, {
                {"Perspective", (uint32_t)CameraType::Perspective},
                {"Equirectangular", (uint32_t)CameraType::Equirectangular}
            }
        },
    };
    static std::map<std::string, std::map<std::string, uint32_t>> s_enumNameToIntTables;
    static std::map<std::string, std::map<uint32_t, std::string>> s_enumIntToNameTables;



    static void initializeEnumTables() {
        for (auto i : s_enumTables) {
            auto &enumNameToIntMap = s_enumNameToIntTables[i.first];
            auto &enumIntToNameMap = s_enumIntToNameTables[i.first];

            for (auto j : i.second) {
                std::string member = j.first;
                uint32_t value = j.second;
                enumNameToIntMap[member] = value;
                enumIntToNameMap[value] = member;
            }
        }

        s_enumTableInitialized = true;
    }

    uint32_t getNumEnumMembers(const char* typeName) {
        if (!s_enumTableInitialized)
            initializeEnumTables();

        if (s_enumTables.count(typeName) == 0)
            return 0;

        const auto &table = s_enumTables.at(typeName);
        return (uint32_t)table.size();
    }
    
    const char* getEnumMemberAt(const char* typeName, uint32_t index) {
        if (!s_enumTableInitialized)
            initializeEnumTables();

        if (s_enumTables.count(typeName) == 0)
            return nullptr;

        const auto &table = s_enumTables.at(typeName);
        if (index >= table.size())
            return nullptr;

        return table[index].first;
    }

    uint32_t getEnumValueFromMember(const char* typeName, const char* member) {
        if (!s_enumTableInitialized)
            initializeEnumTables();

        if (s_enumNameToIntTables.count(typeName) == 0)
            return 0xFFFFFFFF;

        const auto &table = s_enumNameToIntTables.at(typeName);
        if (table.count(member) == 0)
            return 0xFFFFFFFF;

        return table.at(member);
    }

    const char* getEnumMemberFromValue(const char* typeName, uint32_t value) {
        if (!s_enumTableInitialized)
            initializeEnumTables();

        if (s_enumIntToNameTables.count(typeName) == 0)
            return nullptr;

        const auto &table = s_enumIntToNameTables.at(typeName);
        if (table.count(value) == 0)
            return nullptr;

        return table.at(value).c_str();
    }
}
