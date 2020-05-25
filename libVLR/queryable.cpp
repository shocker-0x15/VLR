#include "queryable.h"

namespace VLR {
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
#define VLR_DICT_ITEM(str, e) { str, static_cast<uint32_t>(e) }
    static const std::map<std::string, std::vector<std::pair<const char*, uint32_t>>> s_enumTables = {
        {
            EnumSpectrumType, {
                VLR_DICT_ITEM("Reflectance", SpectrumType::Reflectance),
                VLR_DICT_ITEM("Transmittance", SpectrumType::Transmittance),
                VLR_DICT_ITEM("Light Source", SpectrumType::LightSource),
                VLR_DICT_ITEM("NA", SpectrumType::NA),
            }
        },
        {
            EnumColorSpace, {
                VLR_DICT_ITEM("Rec709(D65) sRGB Gamma", ColorSpace::Rec709_D65_sRGBGamma),
                VLR_DICT_ITEM("Rec709(D65)", ColorSpace::Rec709_D65),
                VLR_DICT_ITEM("XYZ", ColorSpace::XYZ),
                VLR_DICT_ITEM("xyY", ColorSpace::xyY),
            }
        },
        {
            EnumDataFormat, {
                VLR_DICT_ITEM("RGB8x3", DataFormat::RGB8x3),
                VLR_DICT_ITEM("RGB_8x4", DataFormat::RGB_8x4),
                VLR_DICT_ITEM("RGBA8x4", DataFormat::RGBA8x4),
                VLR_DICT_ITEM("RGBA16Fx4", DataFormat::RGBA16Fx4),
                VLR_DICT_ITEM("RGBA32Fx4", DataFormat::RGBA32Fx4),
                VLR_DICT_ITEM("RG32Fx2", DataFormat::RG32Fx2),
                VLR_DICT_ITEM("Gray32F", DataFormat::Gray32F),
                VLR_DICT_ITEM("Gray8", DataFormat::Gray8),
                VLR_DICT_ITEM("GrayA8x2", DataFormat::GrayA8x2),
                VLR_DICT_ITEM("BC1", DataFormat::BC1),
                VLR_DICT_ITEM("BC2", DataFormat::BC2),
                VLR_DICT_ITEM("BC3", DataFormat::BC3),
                VLR_DICT_ITEM("BC4", DataFormat::BC4),
                VLR_DICT_ITEM("BC4_Signed", DataFormat::BC4_Signed),
                VLR_DICT_ITEM("BC5", DataFormat::BC5),
                VLR_DICT_ITEM("BC5_Signed", DataFormat::BC5_Signed),
                VLR_DICT_ITEM("BC6H", DataFormat::BC6H),
                VLR_DICT_ITEM("BC6H_Signed", DataFormat::BC6H_Signed),
                VLR_DICT_ITEM("BC7", DataFormat::BC7),
            }
        },
        {
            EnumBumpType, {
                VLR_DICT_ITEM("Normal Map (DirectX)", BumpType::NormalMap_DirectX),
                VLR_DICT_ITEM("Normal Map (OpenGL)", BumpType::NormalMap_OpenGL),
                VLR_DICT_ITEM("Height Map", BumpType::HeightMap),
            }
        },
        {
            EnumTextureFilter, {
                VLR_DICT_ITEM("Nearest", TextureFilter::Nearest),
                VLR_DICT_ITEM("Linear", TextureFilter::Linear),
            }
        },
        {
            EnumTextureWrapMode, {
                VLR_DICT_ITEM("Repeat", TextureWrapMode::Repeat),
                VLR_DICT_ITEM("Clamp to Edge", TextureWrapMode::ClampToEdge),
                VLR_DICT_ITEM("Mirror", TextureWrapMode::Mirror),
                VLR_DICT_ITEM("Clamp to Border", TextureWrapMode::ClampToBorder),
            }
        },
        {
            EnumTangentType, {
                VLR_DICT_ITEM("TC0 Direction", TangentType::TC0Direction),
                VLR_DICT_ITEM("Radial X", TangentType::RadialX),
                VLR_DICT_ITEM("Radial Y", TangentType::RadialY),
                VLR_DICT_ITEM("Radial Z", TangentType::RadialZ),
            }
        },
    };
#undef VLR_DICT_ITEM
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
        return static_cast<uint32_t>(table.size());
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
        return static_cast<EnumType>(0xFFFFFFFF);
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
            return static_cast<EnumType>(0xFFFFFFFF); \
 \
        const auto& table = s_enumNameToIntTables.at(Enum ## EnumType); \
        if (table.count(member) == 0) \
            return static_cast<EnumType>(0xFFFFFFFF); \
 \
        return static_cast<EnumType>(table.at(member)); \
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
        if (table.count(static_cast<uint32_t>(value)) == 0) \
            return nullptr; \
 \
        return table.at(static_cast<uint32_t>(value)).c_str(); \
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
