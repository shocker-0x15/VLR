#include "queryable.h"

namespace vlr {
    bool testParamName(const std::string& paramNameA, const std::string& paramNameB) {
        return tolower(paramNameA) == tolower(paramNameB);
    }



    const char* ParameterFloat = "Float";
    const char* ParameterPoint3D = "Point3D";
    const char* ParameterVector3D = "Vector3D";
    const char* ParameterNormal3D = "Normal3D";
    const char* ParameterQuaternion = "Quaternion";
    const char* ParameterSpectrum = "Spectrum";
    const char* ParameterAlpha = "Alpha";
    const char* ParameterTextureCoordinates = "TextureCoordinates";

    const char* ParameterImage = "Image";
    const char* ParameterSurfaceMaterial = "SurfaceMaterial";

    const char* EnumSpectrumType = "SpectrumType";
    const char* EnumColorSpace = "ColorSpace";
    const char* EnumDataFormat = "DataFormat";
    const char* EnumBumpType = "BumpType";
    const char* EnumTextureFilter = "TextureFilter";
    const char* EnumTextureWrapMode = "TextureWrapMode";
    const char* EnumTangentType = "TangentType";

    struct EnumNameComparator {
        bool operator()(const std::string& strA, const std::string& strB) const {
            return tolower(strA) < tolower(strB);
        }
    };

    static bool s_enumTableInitialized = false;
    static const std::map<std::string, std::vector<std::pair<const char*, uint32_t>>> s_enumTables = {
        {
            EnumSpectrumType, {
                {"Reflectance", (uint32_t)SpectrumType::Reflectance},
                {"Transmittance", (uint32_t)SpectrumType::Transmittance},
                {"Light Source", (uint32_t)SpectrumType::LightSource},
                {"NA", (uint32_t)SpectrumType::NA},
            }
        },
        {
            EnumColorSpace, {
                {"Rec709(D65) sRGB Gamma", (uint32_t)ColorSpace::Rec709_D65_sRGBGamma},
                {"Rec709(D65)", (uint32_t)ColorSpace::Rec709_D65},
                {"XYZ", (uint32_t)ColorSpace::XYZ},
                {"xyY", (uint32_t)ColorSpace::xyY},
            }
        },
        {
            EnumDataFormat, {
                {"RGB8x3", (uint32_t)DataFormat::RGB8x3},
                {"RGB_8x4", (uint32_t)DataFormat::RGB_8x4},
                {"RGBA8x4", (uint32_t)DataFormat::RGBA8x4},
                {"RGBA16Fx4", (uint32_t)DataFormat::RGBA16Fx4},
                {"RGBA32Fx4", (uint32_t)DataFormat::RGBA32Fx4},
                {"RG32Fx2", (uint32_t)DataFormat::RG32Fx2},
                {"Gray32F", (uint32_t)DataFormat::Gray32F},
                {"Gray8", (uint32_t)DataFormat::Gray8},
                {"GrayA8x2", (uint32_t)DataFormat::GrayA8x2},
                {"BC1", (uint32_t)DataFormat::BC1},
                {"BC2", (uint32_t)DataFormat::BC2},
                {"BC3", (uint32_t)DataFormat::BC3},
                {"BC4", (uint32_t)DataFormat::BC4},
                {"BC4_Signed", (uint32_t)DataFormat::BC4_Signed},
                {"BC5", (uint32_t)DataFormat::BC5},
                {"BC5_Signed", (uint32_t)DataFormat::BC5_Signed},
                {"BC6H", (uint32_t)DataFormat::BC6H},
                {"BC6H_Signed", (uint32_t)DataFormat::BC6H_Signed},
                {"BC7", (uint32_t)DataFormat::BC7},
            }
        },
        {
            EnumBumpType, {
                {"Normal Map (DirectX)", (uint32_t)BumpType::NormalMap_DirectX},
                {"Normal Map (OpenGL)", (uint32_t)BumpType::NormalMap_OpenGL},
                {"Height Map", (uint32_t)BumpType::HeightMap},
            }
        },
        {
            EnumTextureFilter, {
                {"Nearest", (uint32_t)TextureFilter::Nearest},
                {"Linear", (uint32_t)TextureFilter::Linear},
                {"None", (uint32_t)TextureFilter::None},
            }
        },
        {
            EnumTextureWrapMode, {
                {"Repeat", (uint32_t)TextureWrapMode::Repeat},
                {"Clamp to Edge", (uint32_t)TextureWrapMode::ClampToEdge},
                {"Mirror", (uint32_t)TextureWrapMode::Mirror},
                {"Clamp to Border", (uint32_t)TextureWrapMode::ClampToBorder},
            }
        },
        {
            EnumTangentType, {
                {"TC0 Direction", (uint32_t)TangentType::TC0Direction},
                {"Radial X", (uint32_t)TangentType::RadialX},
                {"Radial Y", (uint32_t)TangentType::RadialY},
                {"Radial Z", (uint32_t)TangentType::RadialZ},
            }
        },
    };
    static std::map<std::string, std::map<std::string, uint32_t, EnumNameComparator>> s_enumNameToIntTables;
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

    template <typename EnumType>
    EnumType getEnumValueFromMember(const char* member) {
        VLRAssert_ShouldNotBeCalled();
        return (EnumType)0xFFFFFFFF;
    }
    template <typename EnumType>
    const char* getEnumMemberFromValue(EnumType value) {
        VLRAssert_ShouldNotBeCalled();
        return nullptr;
    }

#define VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(EnumType) \
    template <> \
    EnumType getEnumValueFromMember(const char* member) { \
        if (!s_enumTableInitialized) \
            initializeEnumTables(); \
 \
        if (s_enumNameToIntTables.count(Enum ## EnumType) == 0) \
            return (EnumType)0xFFFFFFFF; \
 \
        const auto& table = s_enumNameToIntTables.at(Enum ## EnumType); \
        if (table.count(member) == 0) \
            return (EnumType)0xFFFFFFFF; \
 \
        return (EnumType)table.at(member); \
    } \
    template EnumType getEnumValueFromMember<EnumType>(const char* member)

#define VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(EnumType) \
    template <> \
    const char* getEnumMemberFromValue(EnumType value) { \
        if (!s_enumTableInitialized) \
            initializeEnumTables(); \
 \
        if (s_enumIntToNameTables.count(Enum ## EnumType) == 0) \
            return nullptr; \
 \
        const auto& table = s_enumIntToNameTables.at(Enum ## EnumType); \
        if (table.count((uint32_t)value) == 0) \
            return nullptr; \
 \
        return table.at((uint32_t)value).c_str(); \
    } \
    template const char* getEnumMemberFromValue<EnumType>(EnumType value)

    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(SpectrumType);
    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(ColorSpace);
    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(DataFormat);
    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(BumpType);
    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(TextureFilter);
    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(TextureWrapMode);
    VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER(TangentType);

    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(SpectrumType);
    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(ColorSpace);
    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(DataFormat);
    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(BumpType);
    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(TextureFilter);
    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(TextureWrapMode);
    VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE(TangentType);

#undef VLR_DEFINE_GET_ENUM_MEMBER_FROM_VALUE
#undef VLR_DEFINE_GET_ENUM_VALUE_FROM_MEMBER
}
