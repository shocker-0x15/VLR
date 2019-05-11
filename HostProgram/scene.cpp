#include "scene.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>



namespace DDS {
    enum class Format : uint32_t {
        BC1_UNorm = 71,
        BC1_UNorm_sRGB = 72,
        BC2_UNorm = 74,
        BC2_UNorm_sRGB = 75,
        BC3_UNorm = 77,
        BC3_UNorm_sRGB = 78,
        BC4_UNorm = 80,
        BC4_SNorm = 81,
        BC5_UNorm = 83,
        BC5_SNorm = 84,
        BC6H_UF16 = 95,
        BC6H_SF16 = 96,
        BC7_UNorm = 98,
        BC7_UNorm_sRGB = 99,
    };

    struct Header {
        struct Flags {
            enum Value : uint32_t {
                Caps = 1 << 0,
                Height = 1 << 1,
                Width = 1 << 2,
                Pitch = 1 << 3,
                PixelFormat = 1 << 12,
                MipMapCount = 1 << 17,
                LinearSize = 1 << 19,
                Depth = 1 << 23
            } value;

            Flags() : value((Value)0) {}
            Flags(Value v) : value(v) {}

            Flags operator&(Flags v) const {
                return (Value)(value & v.value);
            }
            Flags operator|(Flags v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct PFFlags {
            enum Value : uint32_t {
                AlphaPixels = 1 << 0,
                Alpha = 1 << 1,
                FourCC = 1 << 2,
                PaletteIndexed4 = 1 << 3,
                PaletteIndexed8 = 1 << 5,
                RGB = 1 << 6,
                Luminance = 1 << 17,
                BumpDUDV = 1 << 19,
            } value;

            PFFlags() : value((Value)0) {}
            PFFlags(Value v) : value(v) {}

            PFFlags operator&(PFFlags v) const {
                return (Value)(value & v.value);
            }
            PFFlags operator|(PFFlags v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct Caps {
            enum Value : uint32_t {
                Alpha = 1 << 1,
                Complex = 1 << 3,
                Texture = 1 << 12,
                MipMap = 1 << 22,
            } value;

            Caps() : value((Value)0) {}
            Caps(Value v) : value(v) {}

            Caps operator&(Caps v) const {
                return (Value)(value & v.value);
            }
            Caps operator|(Caps v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        struct Caps2 {
            enum Value : uint32_t {
                CubeMap = 1 << 9,
                CubeMapPositiveX = 1 << 10,
                CubeMapNegativeX = 1 << 11,
                CubeMapPositiveY = 1 << 12,
                CubeMapNegativeY = 1 << 13,
                CubeMapPositiveZ = 1 << 14,
                CubeMapNegativeZ = 1 << 15,
                Volume = 1 << 22,
            } value;

            Caps2() : value((Value)0) {}
            Caps2(Value v) : value(v) {}

            Caps2 operator&(Caps2 v) const {
                return (Value)(value & v.value);
            }
            Caps2 operator|(Caps2 v) const {
                return (Value)(value | v.value);
            }
            bool operator==(uint32_t v) const {
                return value == v;
            }
            bool operator!=(uint32_t v) const {
                return value != v;
            }
        };

        uint32_t m_magic;
        uint32_t m_size;
        Flags m_flags;
        uint32_t m_height;
        uint32_t m_width;
        uint32_t m_pitchOrLinearSize;
        uint32_t m_depth;
        uint32_t m_mipmapCount;
        uint32_t m_reserved1[11];
        uint32_t m_PFSize;
        PFFlags m_PFFlags;
        uint32_t m_fourCC;
        uint32_t m_RGBBitCount;
        uint32_t m_RBitMask;
        uint32_t m_GBitMask;
        uint32_t m_BBitMask;
        uint32_t m_RGBAlphaBitMask;
        Caps m_caps;
        Caps2 m_caps2;
        uint32_t m_reservedCaps[2];
        uint32_t m_reserved2;
    };
    static_assert(sizeof(Header) == 128, "sizeof(Header) must be 128.");

    struct HeaderDX10 {
        Format m_format;
        uint32_t m_dimension;
        uint32_t m_miscFlag;
        uint32_t m_arraySize;
        uint32_t m_miscFlag2;
    };
    static_assert(sizeof(HeaderDX10) == 20, "sizeof(HeaderDX10) must be 20.");

    static uint8_t** load(const char* filepath, int32_t* width, int32_t* height, int32_t* mipCount, size_t** sizes, Format* format) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            hpprintf("Not found: %s\n", filepath);
            return nullptr;
        }

        ifs.seekg(0, std::ios::end);
        size_t fileSize = ifs.tellg();

        ifs.clear();
        ifs.seekg(0, std::ios::beg);

        Header header;
        ifs.read((char*)&header, sizeof(Header));
        if (header.m_magic != 0x20534444 || header.m_fourCC != 0x30315844) {
            hpprintf("Non dds (dx10) file: %s", filepath);
            return nullptr;
        }

        HeaderDX10 dx10Header;
        ifs.read((char*)&dx10Header, sizeof(HeaderDX10));

        *width = header.m_width;
        *height = header.m_height;
        *format = (Format)dx10Header.m_format;

        if (*format != Format::BC1_UNorm && *format != Format::BC1_UNorm_sRGB &&
            *format != Format::BC2_UNorm && *format != Format::BC2_UNorm_sRGB &&
            *format != Format::BC3_UNorm && *format != Format::BC3_UNorm_sRGB &&
            *format != Format::BC4_UNorm && *format != Format::BC4_SNorm &&
            *format != Format::BC5_UNorm && *format != Format::BC5_SNorm &&
            *format != Format::BC6H_UF16 && *format != Format::BC6H_SF16 &&
            *format != Format::BC7_UNorm && *format != Format::BC7_UNorm_sRGB) {
            hpprintf("No support for non block compressed formats: %s", filepath);
            return nullptr;
        }

        const size_t dataSize = fileSize - (sizeof(Header) + sizeof(HeaderDX10));

        *mipCount = 1;
        if ((header.m_flags & Header::Flags::MipMapCount) != 0)
            *mipCount = header.m_mipmapCount;

        uint8_t** data = new uint8_t*[*mipCount];
        *sizes = new size_t[*mipCount];
        int32_t mipWidth = *width;
        int32_t mipHeight = *height;
        uint32_t blockSize = 16;
        if (*format == Format::BC1_UNorm || *format == Format::BC1_UNorm_sRGB ||
            *format == Format::BC4_UNorm || *format == Format::BC4_SNorm)
            blockSize = 8;
        size_t cumDataSize = 0;
        for (int i = 0; i < *mipCount; ++i) {
            int32_t bw = (mipWidth + 3) / 4;
            int32_t bh = (mipHeight + 3) / 4;
            size_t mipDataSize = bw * bh * blockSize;

            data[i] = new uint8_t[mipDataSize];
            (*sizes)[i] = mipDataSize;
            ifs.read((char*)data[i], mipDataSize);
            cumDataSize += mipDataSize;

            mipWidth = std::max<int32_t>(1, mipWidth / 2);
            mipHeight = std::max<int32_t>(1, mipHeight / 2);
        }
        Assert(cumDataSize == dataSize, "Data size mismatch.");

        return data;
    }

    static void free(uint8_t** data, int32_t mipCount, size_t* sizes) {
        for (int i = mipCount - 1; i >= 0; --i)
            delete[] data[i];
        delete[] sizes;
        delete[] data;
    }
}



static std::map<std::tuple<std::string, VLRSpectrumType, VLRColorSpace>, VLRCpp::Image2DRef> s_image2DCache;

// TODO: colorSpace should be determined from read image?
static VLRCpp::Image2DRef loadImage2D(const VLRCpp::ContextRef &context, const std::string &filepath, VLRSpectrumType spectrumType, VLRColorSpace colorSpace) {
    using namespace VLRCpp;
    using namespace VLR;

    Image2DRef ret;

    auto key = std::make_tuple(filepath, spectrumType, colorSpace);
    if (s_image2DCache.count(key))
        return s_image2DCache.at(key);

    hpprintf("Read image: %s...", filepath.c_str());

    bool fileExists = false;
    {
        std::ifstream ifs(filepath);
        fileExists = ifs.is_open();
    }
    if (!fileExists) {
        hpprintf("Not found.\n");
        return ret;
    }

    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

//#define OVERRIDE_BY_DDS

#if defined(OVERRIDE_BY_DDS)
    std::string ddsFilepath = filepath;
    ddsFilepath = filepath.substr(0, filepath.find_last_of('.'));
    ddsFilepath += ".dds";
    {
        std::ifstream ifs(ddsFilepath);
        if (ifs.is_open())
            ext = "dds";
    }
#endif

    if (ext == "exr") {
        using namespace Imf;
        using namespace Imath;
        RgbaInputFile file(filepath.c_str());
        Imf::Header header = file.header();

        Box2i dw = file.dataWindow();
        long width = dw.max.x - dw.min.x + 1;
        long height = dw.max.y - dw.min.y + 1;
        Array2D<Rgba> pixels{ height, width };
        pixels.resizeErase(height, width);
        file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
        file.readPixels(dw.min.y, dw.max.y);

        Rgba* linearImageData = new Rgba[width * height];
        Rgba* curDataHead = linearImageData;
        for (int i = 0; i < height; ++i) {
            std::copy_n(pixels[i], width, (Rgba*)curDataHead);
            for (int j = 0; j < width; ++j) {
                Rgba &pix = curDataHead[j];
                pix.r = pix.r >= 0.0f ? pix.r : (half)0.0f;
                pix.g = pix.g >= 0.0f ? pix.g : (half)0.0f;
                pix.b = pix.b >= 0.0f ? pix.b : (half)0.0f;
                pix.a = pix.a >= 0.0f ? pix.a : (half)0.0f;
            }
            curDataHead += width;
        }

        ret = context->createLinearImage2D((uint8_t*)linearImageData, width, height, VLRDataFormat_RGBA16Fx4, spectrumType, colorSpace);

        delete[] linearImageData;
    }
    else if (ext == "dds") {
        int32_t width, height, mipCount;
        size_t* sizes;
        DDS::Format format;
#if defined(OVERRIDE_BY_DDS)
        uint8_t** data = DDS::load(ddsFilepath.c_str(), &width, &height, &mipCount, &sizes, &format);
#else
        uint8_t** data = DDS::load(filepath.c_str(), &width, &height, &mipCount, &sizes, &format);
#endif

        const auto translate = [](DDS::Format ddsFormat, VLRDataFormat* vlrFormat, bool* needsDegamma) {
            *needsDegamma = false;
            switch (ddsFormat) {
            case DDS::Format::BC1_UNorm:
                *vlrFormat = VLRDataFormat_BC1;
                break;
            case DDS::Format::BC1_UNorm_sRGB:
                *vlrFormat = VLRDataFormat_BC1;
                *needsDegamma = true;
                break;
            case DDS::Format::BC2_UNorm:
                *vlrFormat = VLRDataFormat_BC2;
                break;
            case DDS::Format::BC2_UNorm_sRGB:
                *vlrFormat = VLRDataFormat_BC2;
                *needsDegamma = true;
                break;
            case DDS::Format::BC3_UNorm:
                *vlrFormat = VLRDataFormat_BC3;
                break;
            case DDS::Format::BC3_UNorm_sRGB:
                *vlrFormat = VLRDataFormat_BC3;
                *needsDegamma = true;
                break;
            case DDS::Format::BC4_UNorm:
                *vlrFormat = VLRDataFormat_BC4;
                break;
            case DDS::Format::BC4_SNorm:
                *vlrFormat = VLRDataFormat_BC4_Signed;
                break;
            case DDS::Format::BC5_UNorm:
                *vlrFormat = VLRDataFormat_BC5;
                break;
            case DDS::Format::BC5_SNorm:
                *vlrFormat = VLRDataFormat_BC5_Signed;
                break;
            case DDS::Format::BC6H_UF16:
                *vlrFormat = VLRDataFormat_BC6H;
                break;
            case DDS::Format::BC6H_SF16:
                *vlrFormat = VLRDataFormat_BC6H_Signed;
                break;
            case DDS::Format::BC7_UNorm:
                *vlrFormat = VLRDataFormat_BC7;
                break;
            case DDS::Format::BC7_UNorm_sRGB:
                *vlrFormat = VLRDataFormat_BC7;
                *needsDegamma = true;
                break;
            default:
                break;
            }
        };

        VLRDataFormat vlrFormat;
        bool needsDegamma;
        translate(format, &vlrFormat, &needsDegamma);

        ret = context->createBlockCompressedImage2D(data, sizes, mipCount, width, height, vlrFormat, spectrumType, colorSpace);
        Assert(ret, "failed to load a block compressed texture.");

        DDS::free(data, mipCount, sizes);
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filepath.c_str(), &width, &height, &n, 0);
        if (n == 4)
            ret = context->createLinearImage2D(linearImageData, width, height, VLRDataFormat_RGBA8x4, spectrumType, colorSpace);
        else if (n == 3)
            ret = context->createLinearImage2D(linearImageData, width, height, VLRDataFormat_RGB8x3, spectrumType, colorSpace);
        else if (n == 2)
            ret = context->createLinearImage2D(linearImageData, width, height, VLRDataFormat_GrayA8x2, spectrumType, colorSpace);
        else if (n == 1)
            ret = context->createLinearImage2D(linearImageData, width, height, VLRDataFormat_Gray8, spectrumType, colorSpace);
        else
            Assert_ShouldNotBeCalled();
        stbi_image_free(linearImageData);
    }

    hpprintf("done.\n");

    s_image2DCache[key] = ret;

    return ret;
}



SurfaceMaterialAttributeTuple createMaterialDefaultFunction(const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
    using namespace VLRCpp;
    using namespace VLR;

    aiReturn ret;
    (void)ret;
    aiString strValue;
    float color[3];

    aiMat->Get(AI_MATKEY_NAME, strValue);
    hpprintf("Material: %s\n", strValue.C_Str());

    MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
    ShaderNodeSocket socketNormal;
    ShaderNodeSocket socketAlpha;

    Image2DRef imgDiffuse;
    Image2DTextureShaderNodeRef texDiffuse;
    Image2DRef imgAlpha;
    Image2DTextureShaderNodeRef texAlpha;
    
    if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
        texDiffuse = context->createImage2DTextureShaderNode();
        imgDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
        texDiffuse->setImage(imgDiffuse);
        mat->setNodeAlbedo(texDiffuse->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
    }
    else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
        mat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, color[0], color[1], color[2]);
    }
    else {
        mat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 1.0f, 0.0f, 1.0f);
    }

    if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
        imgAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_NA, VLRColorSpace_Rec709_D65);
        texAlpha = context->createImage2DTextureShaderNode();
        texAlpha->setImage(imgAlpha);
    }

    if (imgAlpha) {
        if (imgAlpha->getOriginalDataFormat() == VLRDataFormat_Gray8)
            socketAlpha = texAlpha->getSocket(VLRShaderNodeSocketType_float, 0);
        else
            socketAlpha = texAlpha->getSocket(VLRShaderNodeSocketType_Alpha, 0);
    }
    else if (imgDiffuse && imgDiffuse->originalHasAlpha()) {
        socketAlpha = texDiffuse->getSocket(VLRShaderNodeSocketType_Alpha, 0);
    }

    return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
}

MeshAttributeTuple perMeshDefaultFunction(const aiMesh* mesh) {
    return MeshAttributeTuple(true, VLRTangentType_TC0Direction);
}

void recursiveConstruct(const VLRCpp::ContextRef &context, const aiScene* objSrc, const aiNode* nodeSrc,
                        const std::vector<SurfaceMaterialAttributeTuple> &matAttrTuples, const PerMeshFunction &meshFunc,
                        VLRCpp::InternalNodeRef* nodeOut) {
    using namespace VLRCpp;
    using namespace VLR;

    if (nodeSrc->mNumMeshes == 0 && nodeSrc->mNumChildren == 0) {
        nodeOut = nullptr;
        return;
    }

    const aiMatrix4x4 &tf = nodeSrc->mTransformation;
    float tfElems[] = {
        tf.a1, tf.a2, tf.a3, tf.a4,
        tf.b1, tf.b2, tf.b3, tf.b4,
        tf.c1, tf.c2, tf.c3, tf.c4,
        tf.d1, tf.d2, tf.d3, tf.d4,
    };

    *nodeOut = context->createInternalNode(nodeSrc->mName.C_Str(), context->createStaticTransform(tfElems));

    std::vector<uint32_t> meshIndices;
    for (int m = 0; m < nodeSrc->mNumMeshes; ++m) {
        const aiMesh* mesh = objSrc->mMeshes[nodeSrc->mMeshes[m]];
        if (mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            hpprintf("ignored non triangle mesh: %s.\n", mesh->mName.C_Str());
            continue;
        }
        hpprintf("Mesh: %s\n", mesh->mName.C_Str());

        MeshAttributeTuple meshAttr = meshFunc(mesh);
        if (!meshAttr.visible)
            continue;

        auto surfMesh = context->createTriangleMeshSurfaceNode(mesh->mName.C_Str());
        const SurfaceMaterialAttributeTuple attrTuple = matAttrTuples[mesh->mMaterialIndex];
        const SurfaceMaterialRef &surfMat = attrTuple.material;
        const ShaderNodeSocket &nodeNormal = attrTuple.nodeNormal;
        const ShaderNodeSocket &nodeAlpha = attrTuple.nodeAlpha;

        std::vector<Vertex> vertices;
        for (int v = 0; v < mesh->mNumVertices; ++v) {
            const aiVector3D &p = mesh->mVertices[v];
            const aiVector3D &n = mesh->mNormals[v];
            Vector3D tangent, bitangent;
            if (mesh->mTangents == nullptr)
                Normal3D(n.x, n.y, n.z).makeCoordinateSystem(&tangent, &bitangent);
            const aiVector3D &t = mesh->mTangents ? mesh->mTangents[v] : aiVector3D(tangent[0], tangent[1], tangent[2]);
            const aiVector3D &uv = mesh->mNumUVComponents[0] > 0 ? mesh->mTextureCoords[0][v] : aiVector3D(0, 0, 0);

            Vertex outVtx{ Point3D(p.x, p.y, p.z), Normal3D(n.x, n.y, n.z), Vector3D(t.x, t.y, t.z), TexCoord2D(uv.x, uv.y) };
            float dotNT = dot(outVtx.normal, outVtx.tc0Direction);
            if (std::fabs(dotNT) >= 0.01f)
                outVtx.tc0Direction = normalize(outVtx.tc0Direction - dotNT * outVtx.normal);
            //VLRAssert(absDot(outVtx.normal, outVtx.tc0Direction) < 0.01f, "shading normal and tangent must be orthogonal: %g", absDot(outVtx.normal, outVtx.tangent));
            vertices.push_back(outVtx);
        }
        surfMesh->setVertices(vertices.data(), vertices.size());

        meshIndices.clear();
        for (int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace &face = mesh->mFaces[f];
            meshIndices.push_back(face.mIndices[0]);
            meshIndices.push_back(face.mIndices[1]);
            meshIndices.push_back(face.mIndices[2]);
        }
        surfMesh->addMaterialGroup(meshIndices.data(), meshIndices.size(), surfMat, nodeNormal, nodeAlpha, meshAttr.tangentType);

        (*nodeOut)->addChild(surfMesh);
    }

    if (nodeSrc->mNumChildren) {
        for (int c = 0; c < nodeSrc->mNumChildren; ++c) {
            InternalNodeRef subNode;
            recursiveConstruct(context, objSrc, nodeSrc->mChildren[c], matAttrTuples, meshFunc, &subNode);
            if (subNode != nullptr)
                (*nodeOut)->addChild(subNode);
        }
    }
}

void construct(const VLRCpp::ContextRef &context, const std::string &filePath, bool flipWinding, bool flipV, VLRCpp::InternalNodeRef* nodeOut,
               CreateMaterialFunction matFunc, PerMeshFunction meshFunc) {
    using namespace VLRCpp;
    using namespace VLR;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath, 
        aiProcess_Triangulate |
        aiProcess_CalcTangentSpace |
        (flipWinding ? aiProcess_FlipWindingOrder : 0) |
        (flipV ? aiProcess_FlipUVs : 0));
    if (!scene) {
        hpprintf("Failed to load %s.\n", filePath.c_str());
        *nodeOut = nullptr;
        return;
    }
    hpprintf("Reading: %s done.\n", filePath.c_str());

    std::string pathPrefix = filePath.substr(0, filePath.find_last_of("/") + 1);

    // create materials
    std::vector<SurfaceMaterialAttributeTuple> attrTuples;
    for (int m = 0; m < scene->mNumMaterials; ++m) {
        const aiMaterial* aiMat = scene->mMaterials[m];
        attrTuples.push_back(matFunc(context, aiMat, pathPrefix));
    }

    recursiveConstruct(context, scene, scene->mRootNode, attrTuples, meshFunc, nodeOut);

    hpprintf("Constructing: %s done.\n", filePath.c_str());
}



#define ASSETS_DIR "resources/assets/"

void createCornellBoxScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    auto cornellBox = context->createTriangleMeshSurfaceNode("CornellBox");
    {
        std::vector<Vertex> vertices;

        // Floor
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 5.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f,  1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f,  1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(5.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f, -1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(5.0f, 5.0f) });
        // Back wall
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D(0,  0, 1), Vector3D(1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f, -1.5f), Normal3D(0,  0, 1), Vector3D(1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  3.0f, -1.5f), Normal3D(0,  0, 1), Vector3D(1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f, -1.5f), Normal3D(0,  0, 1), Vector3D(1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        // Ceiling
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f, -1.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  3.0f, -1.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  3.0f,  1.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f,  1.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        // Left wall
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f,  1.5f), Normal3D(1,  0, 0), Vector3D(0,  0, -1), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D(1,  0, 0), Vector3D(0,  0, -1), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f, -1.5f), Normal3D(1,  0, 0), Vector3D(0,  0, -1), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f,  1.5f), Normal3D(1,  0, 0), Vector3D(0,  0, -1), TexCoord2D(0.0f, 0.0f) });
        // Right wall
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f, -1.5f), Normal3D(-1,  0, 0), Vector3D(0,  0,  1), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f,  1.5f), Normal3D(-1,  0, 0), Vector3D(0,  0,  1), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  3.0f,  1.5f), Normal3D(-1,  0, 0), Vector3D(0,  0,  1), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  3.0f, -1.5f), Normal3D(-1,  0, 0), Vector3D(0,  0,  1), TexCoord2D(0.0f, 0.0f) });
        // Light
        vertices.push_back(Vertex{ Point3D(-0.5f,  2.9f, -0.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f,  2.9f, -0.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f,  2.9f,  0.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f,  2.9f,  0.5f), Normal3D(0, -1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        // Light 2
        vertices.push_back(Vertex{ Point3D(0.5f, 0.01f,  1.0f), Normal3D(0,  1,  0), Vector3D(-1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.01f,  1.0f), Normal3D(0,  1,  0), Vector3D(-1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.01f, 1.25f), Normal3D(0,  1,  0), Vector3D(-1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.01f, 1.25f), Normal3D(0,  1,  0), Vector3D(-1,  0,  0), TexCoord2D(0.0f, 0.0f) });

        cornellBox->setVertices(vertices.data(), vertices.size());

        {
            auto image = loadImage2D(context, "resources/checkerboard_line.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
            auto nodeAlbedo = context->createImage2DTextureShaderNode();
            nodeAlbedo->setImage(image);
            nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
            auto matMatte = context->createMatteSurfaceMaterial();
            matMatte->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }

        {
            auto matMatte = context->createMatteSurfaceMaterial();
            matMatte->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65_sRGBGamma, 0.75f, 0.75f, 0.75f);

            std::vector<uint32_t> matGroup = {
                4, 5, 6, 4, 6, 7,
                8, 9, 10, 8, 10, 11,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }

        {
            auto matMatte = context->createMatteSurfaceMaterial();
            matMatte->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65_sRGBGamma, 0.75f, 0.25f, 0.25f);

            //float value[3] = { 0.06f, 0.02f, 0.02f };
            //Float3TextureRef texEmittance = context->createConstantFloat3Texture(value);
            //SurfaceMaterialRef matMatte = context->createDiffuseEmitterSurfaceMaterial(texEmittance);

            std::vector<uint32_t> matGroup = {
                12, 13, 14, 12, 14, 15,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }

        {
            auto matMatte = context->createMatteSurfaceMaterial();
            matMatte->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65_sRGBGamma, 0.25f, 0.25f, 0.75f);

            std::vector<uint32_t> matGroup = {
                16, 17, 18, 16, 18, 19,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }

        {
            auto matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 30.0f, 30.0f, 30.0f);

            std::vector<uint32_t> matGroup = {
                20, 21, 22, 20, 22, 23,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }

        {
            auto matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 100.0f, 100.0f, 100.0f);

            std::vector<uint32_t> matGroup = {
                24, 25, 26, 24, 26, 27,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    shot->scene->addChild(cornellBox);



    InternalNodeRef sphereNode;
    construct(context, "resources/sphere/sphere.obj", false, false, &sphereNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        //auto matA = context->createSpecularReflectionSurfaceMaterial();
        //matA->setImmediateValueCoeffR(VLRColorSpace_Rec709_D65, 0.999f, 0.999f, 0.999f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 1.27579f, 0.940922f, 0.574879f); // Aluminum
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 7.30257f, 6.33458f, 5.16694f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 0.237698f, 0.734847f, 1.37062f); // Copper
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.44233f, 2.55751f, 2.23429f);
        //matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 0.12481f, 0.468228f, 1.44476f); // Gold
        //matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.32107f, 2.23761f, 1.69196f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 2.91705f, 2.92092f, 2.53253f); // Iron
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.06696f, 2.93804f, 2.7429f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 1.9566f, 1.82777f, 1.46089f); // Lead
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.49593f, 3.38158f, 3.17737f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 1.99144f, 1.5186f, 1.00058f); // Mercury
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 5.25161f, 4.6095f, 3.7646f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 2.32528f, 2.06722f, 1.81479f); // Platinum
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 4.19238f, 3.67941f, 3.06551f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 0.157099f, 0.144013f, 0.134847f); // Silver
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.82431f, 3.1451f, 2.27711f);
        ////matA->setImmediateValueEta(VLRColorSpace_Rec709_D65, 2.71866f, 2.50954f, 2.22767f); // Titanium
        ////matA->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.79521f, 3.40035f, 3.00114f);

        auto matA = context->createSpecularScatteringSurfaceMaterial();
        matA->setImmediateValueCoeff(VLRColorSpace_Rec709_D65, 0.999f, 0.999f, 0.999f);
        matA->setImmediateValueEtaExt(VLRColorSpace_Rec709_D65, 1.00036f, 1.00021f, 1.00071f); // Air
        matA->setImmediateValueEtaInt(VLRColorSpace_Rec709_D65, 2.41174f, 2.42343f, 2.44936f); // Diamond
        //matA->setImmediateValueEtaInt(VLRColorSpace_Rec709, 1.33161f, 1.33331f, 1.33799f); // Water
        //matA->setImmediateValueEtaInt(VLRColorSpace_Rec709, 1.51455f, 1.51816f, 1.52642f); // Glass BK7

        //auto matB = context->createDiffuseEmitterSurfaceMaterial();
        //matB->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 1, 1, 1);

        //auto matB = context->createMatteSurfaceMaterial();
        //matB->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.05f, 0.3f, 0.05f);

        //auto mat = context->createMultiSurfaceMaterial();
        //mat->setSubMaterial(0, matA);
        //mat->setSubMaterial(1, matB);

        return SurfaceMaterialAttributeTuple(matA, ShaderNodeSocket(), ShaderNodeSocket());
    });
    shot->scene->addChild(sphereNode);
    sphereNode->setTransform(context->createStaticTransform(scale(0.5f) * translate<float>(0.0f, 1.0f, 0.0f)));



    //Image2DRef imgEnv = loadImage2D(context, "resources/environments/WhiteOne.exr");
    //Float3TextureRef texEnv = context->createImageFloat3Texture(imgEnv);
    //EnvironmentEmitterSurfaceMaterialRef matEnv = context->createEnvironmentEmitterSurfaceMaterial(texEnv);
    //scene->setEnvironment(matEnv);

    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;
    shot->environmentRotation = 0.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0, 1.5f, 6.0f));
        camera->setOrientation(qRotateY<float>(M_PI));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createMaterialTestScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createScaleAndOffsetUVTextureMap2DShaderNode();
        nodeTexCoord->setValues(offset, scale);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);

        Image2DTextureShaderNodeRef nodeAlbedo = context->createImage2DTextureShaderNode();
        nodeAlbedo->setImage(image);
        nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
        nodeAlbedo->setNodeTexCoord(nodeTexCoord->getSocket(VLRShaderNodeSocketType_TextureCoordinates, 0));

        MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
        mat->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
    });
    shot->scene->addChild(modelNode);



    auto light = context->createTriangleMeshSurfaceNode("light");
    {
        std::vector<Vertex> vertices;

        // Light
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        light->setVertices(vertices.data(), vertices.size());

        {
            auto matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 50.0f, 50.0f, 50.0f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, -3.0f) * rotateX<float>(M_PI / 2)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, "resources/material_test/mitsuba_knob.obj", false, false, &modelNode, 
              [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        if (strcmp(strValue.C_Str(), "Base") == 0) {
            MatteSurfaceMaterialRef matteMat = context->createMatteSurfaceMaterial();
            matteMat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.18f, 0.18f, 0.18f);

            mat = matteMat;
        }
        else if (strcmp(strValue.C_Str(), "Glossy") == 0) {
            Image2DRef imgBaseColorAlpha = loadImage2D(context, pathPrefix + "TexturesCom_Leaves0165_1_alphamasked_S.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
            Image2DTextureShaderNodeRef nodeBaseColorAlpha = context->createImage2DTextureShaderNode();
            nodeBaseColorAlpha->setImage(imgBaseColorAlpha);

            UE4SurfaceMaterialRef ue4Mat = context->createUE4SurfaceMaterial();
            ue4Mat->setNodeBaseColor(nodeBaseColorAlpha->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            ue4Mat->setImmediateValueBaseColor(VLRColorSpace_Rec709_D65_sRGBGamma, 0.75f, 0.5f, 0.0025f);
            ue4Mat->setImmediateValueOcclusion(0.0f);
            ue4Mat->setImmediateValueRoughness(0.3f);
            ue4Mat->setImmediateValueMetallic(0.0f);

            mat = ue4Mat;

            socketAlpha = nodeBaseColorAlpha->getSocket(VLRShaderNodeSocketType_Alpha, 0);

            //auto matteMat = context->createMatteSurfaceMaterial();
            //float lambdas[] = { 360.0, 418.75, 477.5, 536.25, 595.0, 653.75, 712.5, 771.25, 830.0 };
            //float values[] = { 0.8f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            ////auto nodeAlbedo = context->createRegularSampledSpectrumShaderNode();
            ////nodeAlbedo->setImmediateValueSpectrum(360.0f, 830.0f, values, lengthof(values));
            //auto nodeAlbedo = context->createIrregularSampledSpectrumShaderNode();
            //nodeAlbedo->setImmediateValueSpectrum(lambdas, values, lengthof(values));
            //matteMat->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            //mat = matteMat;

            //MicrofacetReflectionSurfaceMaterialRef mfMat = context->createMicrofacetReflectionSurfaceMaterial();
            //// Aluminum
            //mfMat->setImmediateValueEta(RGBSpectrum(1.27579f, 0.940922f, 0.574879f));
            //mfMat->setImmediateValue_k(RGBSpectrum(7.30257f, 6.33458f, 5.16694f));
            //mfMat->setImmediateValueRoughness(0.2f);
            //mfMat->setImmediateValueAnisotropy(0.9f);
            //mfMat->setImmediateValueRotation(0.0f);

            //mat = mfMat;

            //GeometryShaderNodeRef nodeGeom = context->createGeometryShaderNode();
            //Vector3DToSpectrumShaderNodeRef nodeVec2Sp = context->createVector3DToSpectrumShaderNode();
            //nodeVec2Sp->setNodeVector3D(nodeGeom->getSocket(VLRShaderNodeSocketType_Vector3D, 0));
            //MatteSurfaceMaterialRef mtMat = context->createMatteSurfaceMaterial();
            //mtMat->setNodeAlbedo(nodeVec2Sp->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

            //mat = mtMat;
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    },
              [](const aiMesh* mesh) {
        return MeshAttributeTuple(true, VLRTangentType_RadialY);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.04089, 0)));



    auto imgEnv = loadImage2D(context, "resources/material_test/Chelsea_Stairs_3k.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceTextured(nodeEnvTex);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = -M_PI / 2;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0.0f, 5.0f, 10.0f));
        camera->setOrientation(qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createColorCheckerScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    const float ColorCheckerLambdas[] = {
        380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730
    };
    const float ColorCheckerSpectrumValues[24][36] = {
        { 0.055, 0.058, 0.061, 0.062, 0.062, 0.062, 0.062, 0.062, 0.062, 0.062, 0.062, 0.063, 0.065, 0.070, 0.076, 0.079, 0.081, 0.084, 0.091, 0.103, 0.119, 0.134, 0.143, 0.147, 0.151, 0.158, 0.168, 0.179, 0.188, 0.190, 0.186, 0.181, 0.182, 0.187, 0.196, 0.209 }, // dark skin
        { 0.117, 0.143, 0.175, 0.191, 0.196, 0.199, 0.204, 0.213, 0.228, 0.251, 0.280, 0.309, 0.329, 0.333, 0.315, 0.286, 0.273, 0.276, 0.277, 0.289, 0.339, 0.420, 0.488, 0.525, 0.546, 0.562, 0.578, 0.595, 0.612, 0.625, 0.638, 0.656, 0.678, 0.700, 0.717, 0.734 }, // light skin
        { 0.130, 0.177, 0.251, 0.306, 0.324, 0.330, 0.333, 0.331, 0.323, 0.311, 0.298, 0.285, 0.269, 0.250, 0.231, 0.214, 0.199, 0.185, 0.169, 0.157, 0.149, 0.145, 0.142, 0.141, 0.141, 0.141, 0.143, 0.147, 0.152, 0.154, 0.150, 0.144, 0.136, 0.132, 0.135, 0.147 }, // blue sky
        { 0.051, 0.054, 0.056, 0.057, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.065, 0.067, 0.075, 0.101, 0.145, 0.178, 0.184, 0.170, 0.149, 0.133, 0.122, 0.115, 0.109, 0.105, 0.104, 0.106, 0.109, 0.112, 0.114, 0.114, 0.112, 0.112, 0.115, 0.120, 0.125, 0.130 }, // foliage
        { 0.144, 0.198, 0.294, 0.375, 0.408, 0.421, 0.426, 0.426, 0.419, 0.403, 0.379, 0.346, 0.311, 0.281, 0.254, 0.229, 0.214, 0.208, 0.202, 0.194, 0.193, 0.200, 0.214, 0.230, 0.241, 0.254, 0.279, 0.313, 0.348, 0.366, 0.366, 0.359, 0.358, 0.365, 0.377, 0.398 }, // blue flower
        { 0.136, 0.179, 0.247, 0.297, 0.320, 0.337, 0.355, 0.381, 0.419, 0.466, 0.510, 0.546, 0.567, 0.574, 0.569, 0.551, 0.524, 0.488, 0.445, 0.400, 0.350, 0.299, 0.252, 0.221, 0.204, 0.196, 0.191, 0.188, 0.191, 0.199, 0.212, 0.223, 0.232, 0.233, 0.229, 0.229 }, // bluish green
        { 0.054, 0.054, 0.053, 0.054, 0.054, 0.055, 0.055, 0.055, 0.056, 0.057, 0.058, 0.061, 0.068, 0.089, 0.125, 0.154, 0.174, 0.199, 0.248, 0.335, 0.444, 0.538, 0.587, 0.595, 0.591, 0.587, 0.584, 0.584, 0.590, 0.603, 0.620, 0.639, 0.655, 0.663, 0.663, 0.667 }, // orange
        { 0.122, 0.164, 0.229, 0.286, 0.327, 0.361, 0.388, 0.400, 0.392, 0.362, 0.316, 0.260, 0.209, 0.168, 0.138, 0.117, 0.104, 0.096, 0.090, 0.086, 0.084, 0.084, 0.084, 0.084, 0.084, 0.085, 0.090, 0.098, 0.109, 0.123, 0.143, 0.169, 0.205, 0.244, 0.287, 0.332 }, // purplish blue
        { 0.096, 0.115, 0.131, 0.135, 0.133, 0.132, 0.130, 0.128, 0.125, 0.120, 0.115, 0.110, 0.105, 0.100, 0.095, 0.093, 0.092, 0.093, 0.096, 0.108, 0.156, 0.265, 0.399, 0.500, 0.556, 0.579, 0.588, 0.591, 0.593, 0.594, 0.598, 0.602, 0.607, 0.609, 0.609, 0.610 }, // moderate red
        { 0.092, 0.116, 0.146, 0.169, 0.178, 0.173, 0.158, 0.139, 0.119, 0.101, 0.087, 0.075, 0.066, 0.060, 0.056, 0.053, 0.051, 0.051, 0.052, 0.052, 0.051, 0.052, 0.058, 0.073, 0.096, 0.119, 0.141, 0.166, 0.194, 0.227, 0.265, 0.309, 0.355, 0.396, 0.436, 0.478 }, // purple
        { 0.061, 0.061, 0.062, 0.063, 0.064, 0.066, 0.069, 0.075, 0.085, 0.105, 0.139, 0.192, 0.271, 0.376, 0.476, 0.531, 0.549, 0.546, 0.528, 0.504, 0.471, 0.428, 0.381, 0.347, 0.327, 0.318, 0.312, 0.310, 0.314, 0.327, 0.345, 0.363, 0.376, 0.381, 0.378, 0.379 }, // yellow green
        { 0.063, 0.063, 0.063, 0.064, 0.064, 0.064, 0.065, 0.066, 0.067, 0.068, 0.071, 0.076, 0.087, 0.125, 0.206, 0.305, 0.383, 0.431, 0.469, 0.518, 0.568, 0.607, 0.628, 0.637, 0.640, 0.642, 0.645, 0.648, 0.651, 0.653, 0.657, 0.664, 0.673, 0.680, 0.684, 0.688 }, // orange yellow
        { 0.066, 0.079, 0.102, 0.146, 0.200, 0.244, 0.282, 0.309, 0.308, 0.278, 0.231, 0.178, 0.130, 0.094, 0.070, 0.054, 0.046, 0.042, 0.039, 0.038, 0.038, 0.038, 0.038, 0.039, 0.039, 0.040, 0.041, 0.042, 0.044, 0.045, 0.046, 0.046, 0.048, 0.052, 0.057, 0.065 }, // blue
        { 0.052, 0.053, 0.054, 0.055, 0.057, 0.059, 0.061, 0.066, 0.075, 0.093, 0.125, 0.178, 0.246, 0.307, 0.337, 0.334, 0.317, 0.293, 0.262, 0.230, 0.198, 0.165, 0.135, 0.115, 0.104, 0.098, 0.094, 0.092, 0.093, 0.097, 0.102, 0.108, 0.113, 0.115, 0.114, 0.114 }, // green
        { 0.050, 0.049, 0.048, 0.047, 0.047, 0.047, 0.047, 0.047, 0.046, 0.045, 0.044, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.054, 0.060, 0.072, 0.104, 0.178, 0.312, 0.467, 0.581, 0.644, 0.675, 0.690, 0.698, 0.706, 0.715, 0.724, 0.730, 0.734, 0.738 }, // red
        { 0.058, 0.054, 0.052, 0.052, 0.053, 0.054, 0.056, 0.059, 0.067, 0.081, 0.107, 0.152, 0.225, 0.336, 0.462, 0.559, 0.616, 0.650, 0.672, 0.694, 0.710, 0.723, 0.731, 0.739, 0.746, 0.752, 0.758, 0.764, 0.769, 0.771, 0.776, 0.782, 0.790, 0.796, 0.799, 0.804 }, // yellow
        { 0.145, 0.195, 0.283, 0.346, 0.362, 0.354, 0.334, 0.306, 0.276, 0.248, 0.218, 0.190, 0.168, 0.149, 0.127, 0.107, 0.100, 0.102, 0.104, 0.109, 0.137, 0.200, 0.290, 0.400, 0.516, 0.615, 0.687, 0.732, 0.760, 0.774, 0.783, 0.793, 0.803, 0.812, 0.817, 0.825 }, // magenta
        { 0.108, 0.141, 0.192, 0.236, 0.261, 0.286, 0.317, 0.353, 0.390, 0.426, 0.446, 0.444, 0.423, 0.385, 0.337, 0.283, 0.231, 0.185, 0.146, 0.118, 0.101, 0.090, 0.082, 0.076, 0.074, 0.073, 0.073, 0.074, 0.076, 0.077, 0.076, 0.075, 0.073, 0.072, 0.074, 0.079 }, // cyan
        { 0.189, 0.255, 0.423, 0.660, 0.811, 0.862, 0.877, 0.884, 0.891, 0.896, 0.899, 0.904, 0.907, 0.909, 0.911, 0.910, 0.911, 0.914, 0.913, 0.916, 0.915, 0.916, 0.914, 0.915, 0.918, 0.919, 0.921, 0.923, 0.924, 0.922, 0.922, 0.925, 0.927, 0.930, 0.930, 0.933 }, // white 9.5 (.05 D)
        { 0.171, 0.232, 0.365, 0.507, 0.567, 0.583, 0.588, 0.590, 0.591, 0.590, 0.588, 0.588, 0.589, 0.589, 0.591, 0.590, 0.590, 0.590, 0.589, 0.591, 0.590, 0.590, 0.587, 0.585, 0.583, 0.580, 0.578, 0.576, 0.574, 0.572, 0.571, 0.569, 0.568, 0.568, 0.566, 0.566 }, // neutral 8 (.23 D)
        { 0.144, 0.192, 0.272, 0.331, 0.350, 0.357, 0.361, 0.363, 0.363, 0.361, 0.359, 0.358, 0.358, 0.359, 0.360, 0.360, 0.361, 0.361, 0.360, 0.362, 0.362, 0.361, 0.359, 0.358, 0.355, 0.352, 0.350, 0.348, 0.345, 0.343, 0.340, 0.338, 0.335, 0.334, 0.332, 0.331 }, // neutral 6.5 (.44 D)
        { 0.105, 0.131, 0.163, 0.180, 0.186, 0.190, 0.193, 0.194, 0.194, 0.192, 0.191, 0.191, 0.191, 0.192, 0.192, 0.192, 0.192, 0.192, 0.192, 0.193, 0.192, 0.192, 0.191, 0.189, 0.188, 0.186, 0.184, 0.182, 0.181, 0.179, 0.178, 0.176, 0.174, 0.173, 0.172, 0.171 }, // neutral 5 (.70 D)
        { 0.068, 0.077, 0.084, 0.087, 0.089, 0.090, 0.092, 0.092, 0.091, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.090, 0.089, 0.089, 0.088, 0.087, 0.086, 0.086, 0.085, 0.084, 0.084, 0.083, 0.083, 0.082, 0.081, 0.081, 0.081 }, // neutral 3.5 (1.05 D)
        { 0.031, 0.032, 0.032, 0.033, 0.033, 0.033, 0.033, 0.033, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.033 }, // black 2 (1.5 D)
    };

    auto colorChecker = context->createTriangleMeshSurfaceNode("ColorChecker");
    {
        std::vector<Vertex> vertices;
        for (int i = 0; i < 24; ++i) {
            int32_t x = i % 6;
            int32_t z = i / 6;
            vertices.push_back(Vertex{ Point3D(0.0f + x, 0.0f, 0.0f + z), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
            vertices.push_back(Vertex{ Point3D(0.0f + x, 0.0f, 1.0f + z), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
            vertices.push_back(Vertex{ Point3D(1.0f + x, 0.0f, 1.0f + z), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
            vertices.push_back(Vertex{ Point3D(1.0f + x, 0.0f, 0.0f + z), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });
        }
        colorChecker->setVertices(vertices.data(), vertices.size());

        const float MinLambda = ColorCheckerLambdas[0];
        const float MaxLambda = ColorCheckerLambdas[lengthof(ColorCheckerLambdas) - 1];
        for (int i = 0; i < 24; ++i) {
            auto spectrum = context->createRegularSampledSpectrumShaderNode();
            spectrum->setImmediateValueSpectrum(VLRSpectrumType_Reflectance, MinLambda, MaxLambda, ColorCheckerSpectrumValues[i], lengthof(ColorCheckerLambdas));

            auto matMatte = context->createMatteSurfaceMaterial();
            matMatte->setNodeAlbedo(spectrum->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

            uint32_t indexOffset = 4 * i;
            std::vector<uint32_t> matGroup = {
                indexOffset + 0, indexOffset + 1, indexOffset + 2,
                indexOffset + 0, indexOffset + 2, indexOffset + 3
            };
            colorChecker->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    auto colorCheckerNode = context->createInternalNode("ColorChecker", context->createStaticTransform(translate<float>(-3.0f, 2.0f, 0.0f) * rotateX<float>(M_PI / 2)));
    colorCheckerNode->addChild(colorChecker);
    shot->scene->addChild(colorCheckerNode);

    //TriangleMeshSurfaceNodeRef light = context->createTriangleMeshSurfaceNode("light");
    //{
    //    std::vector<Vertex> vertices;

    //    // Light
    //    vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
    //    vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
    //    vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
    //    vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

    //    light->setVertices(vertices.data(), vertices.size());

    //    {
    //        DiffuseEmitterSurfaceMaterialRef matLight = context->createDiffuseEmitterSurfaceMaterial();
    //        matLight->setImmediateValueEmittance(VLRColorSpace_Rec709, 50.0f, 50.0f, 50.0f);

    //        std::vector<uint32_t> matGroup = {
    //            0, 1, 2, 0, 2, 3
    //        };
    //        light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
    //    }
    //}
    //InternalNodeRef lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, -3.0f) * rotateX<float>(M_PI / 2)));
    //lightNode->addChild(light);
    //shot->scene->addChild(lightNode);

    //const float IlluminantD50SpectrumValues[] = {
    //    23.942, 26.961, 24.488, 29.871, 49.308, 56.513, 60.034, 57.818, 74.825, 87.247,
    //    90.612, 91.368, 95.109, 91.963, 95.724, 96.613, 97.129, 102.099, 100.755, 102.317,
    //    100, 97.735, 98.918, 93.499, 97.688, 99.269, 99.042, 95.722, 98.857, 95.667,
    //    98.19, 103.003, 99.133, 87.381, 91.604, 92.889, 76.854, 86.511, 92.58, 78.23,
    //    57.692, 82.923, 78.274,
    //}; // 360-780[nm]
    //const float* Values = IlluminantD50SpectrumValues;
    //const float MinLambda = 360;
    //const float MaxLambda = 780;
    //const uint32_t NumLambdas = lengthof(IlluminantD50SpectrumValues);
    const float IlluminantD65SpectrumValues[] = {
        46.6383, 52.0891, 49.9755, 54.6482, 82.7549, 91.486, 93.4318, 86.6823, 104.865, 117.008,
        117.812, 114.861, 115.923, 108.811, 109.354, 107.802, 104.79, 107.689, 104.405, 104.046,
        100, 96.3342, 95.788, 88.6856, 90.0062, 89.5991, 87.6987, 83.2886, 83.6992, 80.0268,
        80.2146, 82.2778, 78.2842, 69.7213, 71.6091, 74.349, 61.604, 69.8856, 75.087, 63.5927,
        46.4182, 66.8054, 63.3828, 64.304, 59.4519, 51.959, 57.4406, 60.3125,
    }; // 360-830
    const float* Values = IlluminantD65SpectrumValues;
    const float MinLambda = 360;
    const float MaxLambda = 830;
    const uint32_t NumLambdas = lengthof(IlluminantD65SpectrumValues);
    auto spectrum = context->createRegularSampledSpectrumShaderNode();
    spectrum->setImmediateValueSpectrum(VLRSpectrumType_LightSource, MinLambda, MaxLambda, Values, NumLambdas);
    float envScale = 0.02f;
    //const float IlluminantESpectrumValues[] = { 1.0f, 1.0f };
    //auto spectrum = context->createRegularSampledSpectrumShaderNode();
    //spectrum->setImmediateValueSpectrum(0.0f, 1000.0f, IlluminantESpectrumValues, 2);
    //float envScale = 1.0f;
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceConstant(spectrum);
    //matEnv->setImmediateValueEmittance(VLRColorSpace_xyY, 1.0f / 3, 1.0f / 3, 1.0f);
    matEnv->setImmediateValueScale(envScale);
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0, 0, 6.0f));
        camera->setOrientation(qRotateY<float>(M_PI));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createColorInterpolationTestScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createScaleAndOffsetUVTextureMap2DShaderNode();
        nodeTexCoord->setValues(offset, scale);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);

        Image2DTextureShaderNodeRef nodeAlbedo = context->createImage2DTextureShaderNode();
        nodeAlbedo->setImage(image);
        nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
        nodeAlbedo->setNodeTexCoord(nodeTexCoord->getSocket(VLRShaderNodeSocketType_TextureCoordinates, 0));

        MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
        mat->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
        });
    shot->scene->addChild(modelNode);



    auto colorTestPlate = context->createTriangleMeshSurfaceNode("color_test_plate");
    {
        std::vector<Vertex> vertices;

        // Light
        vertices.push_back(Vertex{ Point3D(-1.0f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.0f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.0f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.0f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        colorTestPlate->setVertices(vertices.data(), vertices.size());

        {
            auto image = loadImage2D(context, "resources/material_test/jumping_colors.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
            auto nodeAlbedo = context->createImage2DTextureShaderNode();
            nodeAlbedo->setImage(image);
            nodeAlbedo->setTextureWrapMode(VLRTextureWrapMode_ClampToEdge, VLRTextureWrapMode_ClampToEdge);
            auto mat = context->createMatteSurfaceMaterial();
            mat->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

            //auto image = loadImage2D(context, "resources/material_test/jumping_colors.png", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65_sRGBGamma);
            //auto nodeEmittance = context->createImage2DTextureShaderNode();
            //nodeEmittance->setImage(image);
            //nodeEmittance->setTextureWrapMode(VLRTextureWrapMode_ClampToEdge, VLRTextureWrapMode_ClampToEdge);
            //auto mat = context->createDiffuseEmitterSurfaceMaterial();
            //mat->setNodeEmittance(nodeEmittance->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            //mat->setImmediateValueScale(10.0f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            colorTestPlate->addMaterialGroup(matGroup.data(), matGroup.size(), mat, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    auto colorTestPlateNode = context->createInternalNode("colorTestPlateNode", context->createStaticTransform(translate<float>(0.0f, 1.5f, 0.0f) * scale(3.0f) * rotateX<float>(M_PI / 4)));
    colorTestPlateNode->addChild(colorTestPlate);
    shot->scene->addChild(colorTestPlateNode);



    //const float IlluminantD50SpectrumValues[] = {
    //    23.942, 26.961, 24.488, 29.871, 49.308, 56.513, 60.034, 57.818, 74.825, 87.247,
    //    90.612, 91.368, 95.109, 91.963, 95.724, 96.613, 97.129, 102.099, 100.755, 102.317,
    //    100, 97.735, 98.918, 93.499, 97.688, 99.269, 99.042, 95.722, 98.857, 95.667,
    //    98.19, 103.003, 99.133, 87.381, 91.604, 92.889, 76.854, 86.511, 92.58, 78.23,
    //    57.692, 82.923, 78.274,
    //}; // 360-780[nm]
    //const float* Values = IlluminantD50SpectrumValues;
    //const float MinLambda = 360;
    //const float MaxLambda = 780;
    //const uint32_t NumLambdas = lengthof(IlluminantD50SpectrumValues);
    const float IlluminantD65SpectrumValues[] = {
        46.6383, 52.0891, 49.9755, 54.6482, 82.7549, 91.486, 93.4318, 86.6823, 104.865, 117.008,
        117.812, 114.861, 115.923, 108.811, 109.354, 107.802, 104.79, 107.689, 104.405, 104.046,
        100, 96.3342, 95.788, 88.6856, 90.0062, 89.5991, 87.6987, 83.2886, 83.6992, 80.0268,
        80.2146, 82.2778, 78.2842, 69.7213, 71.6091, 74.349, 61.604, 69.8856, 75.087, 63.5927,
        46.4182, 66.8054, 63.3828, 64.304, 59.4519, 51.959, 57.4406, 60.3125,
    }; // 360-830
    const float* Values = IlluminantD65SpectrumValues;
    const float MinLambda = 360;
    const float MaxLambda = 830;
    const uint32_t NumLambdas = lengthof(IlluminantD65SpectrumValues);
    auto spectrum = context->createRegularSampledSpectrumShaderNode();
    spectrum->setImmediateValueSpectrum(VLRSpectrumType_LightSource, MinLambda, MaxLambda, Values, NumLambdas);
    float envScale = 0.02f;
    //const float IlluminantESpectrumValues[] = { 1.0f, 1.0f };
    //auto spectrum = context->createRegularSampledSpectrumShaderNode();
    //spectrum->setImmediateValueSpectrum(0.0f, 1000.0f, IlluminantESpectrumValues, 2);
    //float envScale = 1.0f;
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceConstant(spectrum);
    //matEnv->setImmediateValueEmittance(VLRColorSpace_xyY, 1.0f / 3, 1.0f / 3, 1.0f);
    matEnv->setImmediateValueScale(envScale);
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0.0f, 5.0f, 10.0f));
        camera->setOrientation(qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createSubstanceManScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createScaleAndOffsetUVTextureMap2DShaderNode();
        nodeTexCoord->setValues(offset, scale);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);

        Image2DTextureShaderNodeRef nodeAlbedo = context->createImage2DTextureShaderNode();
        nodeAlbedo->setImage(image);
        nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
        nodeAlbedo->setNodeTexCoord(nodeTexCoord->getSocket(VLRShaderNodeSocketType_TextureCoordinates, 0));

        MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
        mat->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
    });
    shot->scene->addChild(modelNode);



    auto light = context->createTriangleMeshSurfaceNode("light");
    {
        std::vector<Vertex> vertices;

        // Light
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        light->setVertices(vertices.data(), vertices.size());

        {
            auto matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 50.0f, 50.0f, 50.0f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, -3.0f) * rotateX<float>(M_PI / 2)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, ASSETS_DIR"spman2/spman2.obj", false, true, &modelNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        if (strcmp(strValue.C_Str(), "_Head1") == 0) {
            auto nodeBaseColor = context->createImage2DTextureShaderNode();
            nodeBaseColor->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_BaseColor.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma));
            auto nodeORM = context->createImage2DTextureShaderNode();
            nodeORM->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_OcclusionRoughnessMetallic.png", VLRSpectrumType_NA, VLRColorSpace_Rec709_D65));
            auto nodeNormal = context->createImage2DTextureShaderNode();
            nodeNormal->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_NormalAlpha.png", VLRSpectrumType_NA, VLRColorSpace_Rec709_D65));

            auto ue4Mat = context->createUE4SurfaceMaterial();
            ue4Mat->setNodeBaseColor(nodeBaseColor->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            ue4Mat->setNodeOcclusionRoughnessMetallic(nodeORM->getSocket(VLRShaderNodeSocketType_float3, 0));

            mat = ue4Mat;
            socketNormal = nodeNormal->getSocket(VLRShaderNodeSocketType_Normal3D, 0);
        }
        else if (strcmp(strValue.C_Str(), "_Body1") == 0) {
            auto nodeBaseColor = context->createImage2DTextureShaderNode();
            nodeBaseColor->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_BaseColor.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma));
            auto nodeORM = context->createImage2DTextureShaderNode();
            nodeORM->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_OcclusionRoughnessMetallic.png", VLRSpectrumType_NA, VLRColorSpace_Rec709_D65));
            auto nodeNormal = context->createImage2DTextureShaderNode();
            nodeNormal->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_NormalAlpha.png", VLRSpectrumType_NA, VLRColorSpace_Rec709_D65));

            auto ue4Mat = context->createUE4SurfaceMaterial();
            ue4Mat->setNodeBaseColor(nodeBaseColor->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            ue4Mat->setNodeOcclusionRoughnessMetallic(nodeORM->getSocket(VLRShaderNodeSocketType_float3, 0));

            mat = ue4Mat;
            socketNormal = nodeNormal->getSocket(VLRShaderNodeSocketType_Normal3D, 0);
        }
        else if (strcmp(strValue.C_Str(), "_Base1") == 0) {
            auto nodeBaseColor = context->createImage2DTextureShaderNode();
            nodeBaseColor->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_BaseColor.png", VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma));
            auto nodeORM = context->createImage2DTextureShaderNode();
            nodeORM->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_OcclusionRoughnessMetallic.png", VLRSpectrumType_NA, VLRColorSpace_Rec709_D65));
            auto nodeNormal = context->createImage2DTextureShaderNode();
            nodeNormal->setImage(loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_NormalAlpha.png", VLRSpectrumType_NA, VLRColorSpace_Rec709_D65));

            auto ue4Mat = context->createUE4SurfaceMaterial();
            ue4Mat->setNodeBaseColor(nodeBaseColor->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            ue4Mat->setNodeOcclusionRoughnessMetallic(nodeORM->getSocket(VLRShaderNodeSocketType_float3, 0));

            mat = ue4Mat;
            socketNormal = nodeNormal->getSocket(VLRShaderNodeSocketType_Normal3D, 0);
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    }, [](const aiMesh* mesh) {
        if (std::strcmp(mesh->mName.C_Str(), "base_base") == 0)
            return MeshAttributeTuple(true, VLRTangentType_RadialY);
        return MeshAttributeTuple(true, VLRTangentType_TC0Direction);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.01, 0) * scale<float>(0.25f)));



    construct(context, "resources/sphere/sphere.obj", false, false, &modelNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        SpecularReflectionSurfaceMaterialRef mat = context->createSpecularReflectionSurfaceMaterial();
        mat->setImmediateValueCoeffR(VLRColorSpace_Rec709_D65_sRGBGamma, 0.999f, 0.999f, 0.999f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 1.27579f, 0.940922f, 0.574879f); // Aluminum
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 7.30257f, 6.33458f, 5.16694f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 0.237698f, 0.734847f, 1.37062f); // Copper
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.44233f, 2.55751f, 2.23429f);
        mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 0.12481f, 0.468228f, 1.44476f); // Gold
        mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.32107f, 2.23761f, 1.69196f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 2.91705f, 2.92092f, 2.53253f); // Iron
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.06696f, 2.93804f, 2.7429f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 1.9566f, 1.82777f, 1.46089f); // Lead
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.49593f, 3.38158f, 3.17737f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 1.99144f, 1.5186f, 1.00058f); // Mercury
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 5.25161f, 4.6095f, 3.7646f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 2.32528f, 2.06722f, 1.81479f); // Platinum
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 4.19238f, 3.67941f, 3.06551f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 0.157099f, 0.144013f, 0.134847f); // Silver
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.82431f, 3.1451f, 2.27711f);
        //mat->setImmediateValueEta(VLRColorSpace_Rec709_D65, 2.71866f, 2.50954f, 2.22767f); // Titanium
        //mat->setImmediateValue_k(VLRColorSpace_Rec709_D65, 3.79521f, 3.40035f, 3.00114f);

        //SpecularScatteringSurfaceMaterialRef mat = context->createSpecularScatteringSurfaceMaterial();
        //mat->setImmediateValueCoeff(VLRColorSpace_Rec709_D65, 0.999f, 0.999f, 0.999f);
        //mat->setImmediateValueEtaExt(VLRColorSpace_Rec709_D65, 1.00036f, 1.00021f, 1.00071f); // Air
        //mat->setImmediateValueEtaInt(VLRColorSpace_Rec709_D65, 2.41174f, 2.42343f, 2.44936f); // Diamond
        ////mat->setImmediateValueEtaInt(VLRColorSpace_Rec709_D65, 1.33161f, 1.33331f, 1.33799f); // Water
        ////mat->setImmediateValueEtaInt(VLRColorSpace_Rec709_D65, 1.51455f, 1.51816f, 1.52642f); // Glass BK7

        //SurfaceMaterialRef mats[] = { matA, matB };
        //SurfaceMaterialRef mat = context->createMultiSurfaceMaterial(mats, lengthof(mats));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(-2.0f, 0.0f, 2.0f) * scale(1.0f) * translate<float>(0.0f, 1.0f, 0.0f)));



    auto imgEnv = loadImage2D(context, "resources/material_test/Chelsea_Stairs_3k.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceTextured(nodeEnvTex);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = -M_PI / 2;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;
    
    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0.0f, 5.0f, 10.0f));
        camera->setOrientation(qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createGalleryScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    auto light = context->createTriangleMeshSurfaceNode("light");
    {
        std::vector<Vertex> vertices;

        // Light
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        light->setVertices(vertices.data(), vertices.size());

        {
            auto matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 50.0f, 50.0f, 50.0f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 2.0f, 0.0f) * rotateX<float>(M_PI)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    const auto gelleryMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            auto matteMat = context->createMatteSurfaceMaterial();
            matteMat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.5f, 0.5f, 0.5f);

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    construct(context, ASSETS_DIR"gallery/gallery.obj", false, true, &modelNode, createMaterialDefaultFunction);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.5f)));



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;
    shot->environmentRotation = 0.0f;
    
    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(-2.3f, 1.0f, 3.5f));
        camera->setOrientation(qRotateY<float>(0.8 * M_PI) * qRotateX<float>(0));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createHairballScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    auto light = context->createTriangleMeshSurfaceNode("light");
    {
        std::vector<Vertex> vertices;

        // Light
        vertices.push_back(Vertex{ Point3D(-1.5f, 0.0f, -1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f, 0.0f, 1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f, 0.0f, 1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f, 0.0f, -1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        light->setVertices(vertices.data(), vertices.size());

        {
            auto matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, 50.0f, 50.0f, 50.0f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, 0.0f) * rotateX<float>(M_PI)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, ASSETS_DIR"hairball/hairball.obj", true, false, &modelNode,
              [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            auto matteMat = context->createMatteSurfaceMaterial();
            matteMat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.5f, 0.5f, 0.5f);

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    },
              [](const aiMesh* mesh) {
        return MeshAttributeTuple(true, VLRTangentType_RadialY);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.1f)));



    shot->renderTargetSizeX = 1024;
    shot->renderTargetSizeY = 1024;

    shot->brightnessCoeff = 1.0f;
    shot->environmentRotation = 0.0f;
    
    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0.0f, 0.0f, 1.5f));
        camera->setOrientation(qRotateY<float>(M_PI) * qRotateX<float>(0 * M_PI / 180));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createRungholtScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    const auto rungholtMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);
        hpprintf("Material: %s\n", strValue.C_Str());

        MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;

        Image2DRef imgDiffuse;
        Image2DTextureShaderNodeRef texDiffuse;
        Image2DRef imgAlpha;
        Image2DTextureShaderNodeRef texAlpha;

        if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
            texDiffuse = context->createImage2DTextureShaderNode();
            imgDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
            texDiffuse->setImage(imgDiffuse);
            texDiffuse->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
            mat->setNodeAlbedo(texDiffuse->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
        }
        else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
            mat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, color[0], color[1], color[2]);
        }
        else {
            mat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 1.0f, 0.0f, 1.0f);
        }

        //if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
        //    imgAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_NA, VLRColorSpace_Rec709_D65);
        //    texAlpha = context->createImage2DTextureShaderNode();
        //    texAlpha->setImage(imgAlpha);
        //}

        /*if (imgAlpha) {
            socketAlpha = texAlpha->getSocket(VLRShaderNodeSocketType_float, 0);
        }
        else*/ if (imgDiffuse && imgDiffuse->originalHasAlpha()) {
            socketAlpha = texDiffuse->getSocket(VLRShaderNodeSocketType_Alpha, 0);
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    construct(context, ASSETS_DIR"rungholt/rungholt.obj", false, true, &modelNode, rungholtMaterialFunc);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.04f)));



    auto ground = context->createTriangleMeshSurfaceNode("ground");
    {
        std::vector<Vertex> vertices;

        const float sizeX = 327.0f;
        const float sizeZ = 275.0f;
        const float y = -0.1f;
        vertices.push_back(Vertex{ Point3D(-sizeX, y, -sizeZ), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-sizeX, y, sizeZ), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(sizeX + 1, y, sizeZ), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(sizeX + 1, y, -sizeZ), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        ground->setVertices(vertices.data(), vertices.size());

        {
            auto mat = context->createMatteSurfaceMaterial();
            mat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.05f, 0.025f, 0.025f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            ground->addMaterialGroup(matGroup.data(), matGroup.size(), mat, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_TC0Direction);
        }
    }
    modelNode->addChild(ground);



    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Playa_Sunrise/Playa_Sunrise.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/Direct_HDR_Capture_of_the_Sun_and_Sky/1400/probe_14-00_latlongmap.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceTextured(nodeEnvTex);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = -0.2 * M_PI;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(10.0f, 5.0f, 0.0f));
        camera->setOrientation(qRotateY<float>(-M_PI / 2 - 0.2 * M_PI) * qRotateX<float>(30 * M_PI / 180));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createPowerplantScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    const auto powerplantMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            auto matteMat = context->createMatteSurfaceMaterial();
            matteMat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.5f, 0.5f, 0.5f);

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    construct(context, ASSETS_DIR"powerplant/powerplant.obj", false, true, &modelNode, createMaterialDefaultFunction);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.0001f)));



    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Barcelona_Rooftops/Barce_Rooftop_C_3k.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceTextured(nodeEnvTex);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(-14.89948f, 1.289585f, -1.764552f));
        camera->setOrientation(Quaternion(-0.089070f, 0.531405f, 0.087888f, 0.837825f));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createAmazonBistroExteriorScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    const auto bistroMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        ai_real floatValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);
        hpprintf("Material: %s\n", strValue.C_Str());
        //if (strcmp(strValue.C_Str(), "Metal_Chrome1") == 0)
        //    printf("");

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            if (aiMat->Get(AI_MATKEY_COLOR_TRANSPARENT, color, nullptr) != aiReturn_SUCCESS) {
                color[0] = color[1] = color[2] = 1;
            }

            if (color[0] < 1 && color[1] < 1 && color[2] < 1) {
                auto glassMat = context->createSpecularScatteringSurfaceMaterial();
                glassMat->setImmediateValueCoeff(VLRColorSpace_Rec709_D65, 1 - color[0], 1 - color[1], 1 - color[2]);
                glassMat->setImmediateValueEtaExt(VLRColorSpace_Rec709_D65, 1.0f, 1.0f, 1.0f);
                glassMat->setImmediateValueEtaInt(VLRColorSpace_Rec709_D65, 1.5f, 1.5f, 1.5f);

                mat = glassMat;
            }
            else {
                auto oldMat = context->createOldStyleSurfaceMaterial();

                oldMat->setImmediateValueGlossiness(0.7f);

                Image2DRef imageDiffuse;
                Image2DRef imageSpecular;
                Image2DRef imageNormal;
                Image2DRef imageAlpha;
                Image2DTextureShaderNodeRef texDiffuse;
                Image2DTextureShaderNodeRef texSpecular;
                Image2DTextureShaderNodeRef texNormal;
                Image2DTextureShaderNodeRef texAlpha;

                if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                    imageDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
                    texDiffuse = context->createImage2DTextureShaderNode();
                    texDiffuse->setImage(imageDiffuse);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
                    imageSpecular = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
                    texSpecular = context->createImage2DTextureShaderNode();
                    texSpecular->setImage(imageSpecular);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS) {
                    imageNormal = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_NA, VLRColorSpace_Rec709_D65);
                    texNormal = context->createImage2DTextureShaderNode();
                    texNormal->setImage(imageNormal);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
                    imageAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_NA, VLRColorSpace_Rec709_D65);
                    texAlpha = context->createImage2DTextureShaderNode();
                    texAlpha->setImage(imageAlpha);
                }

                if (texDiffuse)
                    oldMat->setNodeDiffuseColor(texDiffuse->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

                if (texSpecular)
                    oldMat->setNodeSpecularColor(texSpecular->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

                //if (imageSpecular && imageSpecular->originalHasAlpha())
                //    oldMat->setNodeGlossiness(texSpecular->getSocket(VLRShaderNodeSocketType_Alpha, 0));

                if (texNormal)
                    socketNormal = texNormal->getSocket(VLRShaderNodeSocketType_Normal3D, 0);

                if (texAlpha)
                    socketAlpha = texAlpha->getSocket(VLRShaderNodeSocketType_float, 0);
                else if (imageDiffuse && imageDiffuse->originalHasAlpha())
                    socketAlpha = texDiffuse->getSocket(VLRShaderNodeSocketType_Alpha, 0);

                mat = oldMat;

                if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS) {
                    if (color[0] > 0.0f && color[1] > 0.0f && color[2] > 0.0f) {
                        auto emitter = context->createDiffuseEmitterSurfaceMaterial();
                        emitter->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, color[0], color[1], color[2]);
                        emitter->setImmediateValueScale(30);

                        auto mMat = context->createMultiSurfaceMaterial();
                        mMat->setSubMaterial(0, oldMat);
                        mMat->setSubMaterial(1, emitter);

                        mat = mMat;
                    }
                }
            }
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    const auto grayMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            auto matteMat = context->createMatteSurfaceMaterial();
            matteMat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.5f, 0.5f, 0.5f);

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    construct(context, ASSETS_DIR"Amazon_Bistro/exterior/exterior.obj", false, true, &modelNode, bistroMaterialFunc);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.001f)));



    //Image2DRef imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Barcelona_Rooftops/Barce_Rooftop_C_3k.exr", false);
    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceTextured(nodeEnvTex);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;
    
    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(-0.753442f, 0.140257f, -0.056083f));
        camera->setOrientation(Quaternion(-0.009145f, 0.531434f, -0.005825f, 0.847030f));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.001f;
        camera->setSensitivity(1.0f / (M_PI * lensRadius * lensRadius));
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(lensRadius);
        camera->setObjectPlaneDistance(0.267f);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createEquirectangularCamera();

        camera->setPosition(Point3D(-1.092485f, 0.640749f, -0.094409f));
        camera->setOrientation(Quaternion(0.109960f, 0.671421f, -0.081981f, 0.812352f));

        float phiAngle = 2.127f;
        float thetaAngle = 1.153f;
        camera->setSensitivity(1.0f / (phiAngle * (1 - std::cos(thetaAngle))));
        camera->setAngles(phiAngle, thetaAngle);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(-0.380530f, 0.167073f, -0.309329f));
        camera->setOrientation(Quaternion(0.152768f, 0.422808f, -0.030319f, 0.962553f));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(0.0f);
        camera->setObjectPlaneDistance(1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createAmazonBistroInteriorScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    const auto bistroMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        ai_real floatValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);
        hpprintf("Material: %s\n", strValue.C_Str());
        //if (strcmp(strValue.C_Str(), "Metal_Chrome1") == 0)
        //    printf("");

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            if (aiMat->Get(AI_MATKEY_COLOR_TRANSPARENT, color, nullptr) != aiReturn_SUCCESS) {
                color[0] = color[1] = color[2] = 1;
            }

            if (color[0] < 1 && color[1] < 1 && color[2] < 1) {
                auto glassMat = context->createSpecularScatteringSurfaceMaterial();
                glassMat->setImmediateValueCoeff(VLRColorSpace_Rec709_D65, 1 - color[0], 1 - color[1], 1 - color[2]);
                glassMat->setImmediateValueEtaExt(VLRColorSpace_Rec709_D65, 1.0f, 1.0f, 1.0f);
                glassMat->setImmediateValueEtaInt(VLRColorSpace_Rec709_D65, 1.5f, 1.5f, 1.5f);

                mat = glassMat;
            }
            else {
                auto oldMat = context->createOldStyleSurfaceMaterial();

                oldMat->setImmediateValueGlossiness(0.7f);

                Image2DRef imageDiffuse;
                Image2DRef imageSpecular;
                Image2DRef imageNormal;
                Image2DRef imageAlpha;
                Image2DTextureShaderNodeRef texDiffuse;
                Image2DTextureShaderNodeRef texSpecular;
                Image2DTextureShaderNodeRef texNormal;
                Image2DTextureShaderNodeRef texAlpha;

                if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                    imageDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
                    texDiffuse = context->createImage2DTextureShaderNode();
                    texDiffuse->setImage(imageDiffuse);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
                    imageSpecular = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_Reflectance, VLRColorSpace_Rec709_D65_sRGBGamma);
                    texSpecular = context->createImage2DTextureShaderNode();
                    texSpecular->setImage(imageSpecular);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS) {
                    imageNormal = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_NA, VLRColorSpace_Rec709_D65);
                    texNormal = context->createImage2DTextureShaderNode();
                    texNormal->setImage(imageNormal);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
                    imageAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), VLRSpectrumType_NA, VLRColorSpace_Rec709_D65);
                    texAlpha = context->createImage2DTextureShaderNode();
                    texAlpha->setImage(imageAlpha);
                }

                if (texDiffuse)
                    oldMat->setNodeDiffuseColor(texDiffuse->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

                if (texSpecular)
                    oldMat->setNodeSpecularColor(texSpecular->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

                //if (imageSpecular && imageSpecular->originalHasAlpha())
                //    oldMat->setNodeGlossiness(texSpecular->getSocket(VLRShaderNodeSocketType_Alpha, 0));

                if (texNormal)
                    socketNormal = texNormal->getSocket(VLRShaderNodeSocketType_Normal3D, 0);

                if (texAlpha)
                    socketAlpha = texAlpha->getSocket(VLRShaderNodeSocketType_float, 0);
                else if (imageDiffuse && imageDiffuse->originalHasAlpha())
                    socketAlpha = texDiffuse->getSocket(VLRShaderNodeSocketType_Alpha, 0);

                mat = oldMat;

                if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS) {
                    if (color[0] > 0.0f && color[1] > 0.0f && color[2] > 0.0f) {
                        auto emitter = context->createDiffuseEmitterSurfaceMaterial();
                        emitter->setImmediateValueEmittance(VLRColorSpace_Rec709_D65, color[0], color[1], color[2]);
                        emitter->setImmediateValueScale(30);

                        auto mMat = context->createMultiSurfaceMaterial();
                        mMat->setSubMaterial(0, oldMat);
                        mMat->setSubMaterial(1, emitter);

                        mat = mMat;
                    }
                }
            }
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    const auto grayMaterialFunc = [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodeSocket socketNormal;
        ShaderNodeSocket socketAlpha;
        {
            auto matteMat = context->createMatteSurfaceMaterial();
            matteMat->setImmediateValueAlbedo(VLRColorSpace_Rec709_D65, 0.5f, 0.5f, 0.5f);

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, socketNormal, socketAlpha);
    };
    construct(context, ASSETS_DIR"Amazon_Bistro/Interior/interior_corrected.obj", false, true, &modelNode, bistroMaterialFunc);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.001f)));



    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Barcelona_Rooftops/Barce_Rooftop_C_3k.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/Direct_HDR_Capture_of_the_Sun_and_Sky/1400/probe_14-00_latlongmap.exr", VLRSpectrumType_LightSource, VLRColorSpace_Rec709_D65);
    auto nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    auto matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittanceTextured(nodeEnvTex);
    matEnv->setImmediateValueScale(10);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = 160 * M_PI / 180;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1280;
    shot->renderTargetSizeY = 720;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(-0.177799f, 0.224542f, -0.070547f));
        camera->setOrientation(Quaternion(0.034520f, 0.748582f, -0.032168f, 0.661360f));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.0f;
        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(lensRadius);
        camera->setObjectPlaneDistance(1.000f);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createPerspectiveCamera();

        camera->setPosition(Point3D(0.804731f, 0.146986f, 0.204337f));
        camera->setOrientation(Quaternion(0.101459f, -0.081018f, 0.013512f, 0.991442f));

        camera->setAspectRatio((float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.0001f;
        camera->setSensitivity(1.0f);
        camera->setFovY(40 * M_PI / 180);
        camera->setLensRadius(lensRadius);
        camera->setObjectPlaneDistance(0.036f);

        shot->viewpoints.push_back(camera);
    }
}

void createScene(const VLRCpp::ContextRef &context, Shot* shot) {
    //createCornellBoxScene(context, shot);
    createMaterialTestScene(context, shot);
    //createColorCheckerScene(context, shot);
    //createColorInterpolationTestScene(context, shot);
    //createSubstanceManScene(context, shot);
    //createGalleryScene(context, shot);
    //createHairballScene(context, shot);
    //createRungholtScene(context, shot);
    //createPowerplantScene(context, shot);
    //createAmazonBistroExteriorScene(context, shot);
    //createAmazonBistroInteriorScene(context, shot);
}