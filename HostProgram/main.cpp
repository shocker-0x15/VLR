#include <cstdio>
#include <cstdint>
#include <set>
#include <functional>

#include <VLR/VLRCpp.h>

// only for catching an exception.
#include <optix_world.h>

#include "StopWatch.h"

#define NOMINMAX
#include "imgui.h"
#include "imgui_impl_glfw_gl3.h"
#include "GLToolkit.h"
#include "GLFW/glfw3.h"

// DELETE ME
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfArray.h>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>



#ifdef DEBUG
#   define ENABLE_ASSERT
#endif

#ifdef VLR_Platform_Windows_MSVC
static void debugPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[1024];
    vsprintf_s(str, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#else
#   define debugPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) if (!(expr)) { debugPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); debugPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")

uint64_t g_frameIndex;

struct KeyState {
    uint64_t timesLastChanged[5];
    bool statesLastChanged[5];
    uint32_t lastIndex;

    KeyState() : lastIndex(0) {
        for (int i = 0; i < 5; ++i) {
            timesLastChanged[i] = 0;
            statesLastChanged[i] = false;
        }
    }

    void recordStateChange(bool state, uint64_t time) {
        bool lastState = statesLastChanged[lastIndex];
        if (state == lastState)
            return;

        lastIndex = (lastIndex + 1) % 5;
        statesLastChanged[lastIndex] = !lastState;
        timesLastChanged[lastIndex] = time;
    }

    bool getState(int32_t goBack = 0) const {
        VLRAssert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        VLRAssert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

KeyState g_keyForward;
KeyState g_keyBackward;
KeyState g_keyLeftward;
KeyState g_keyRightward;
KeyState g_keyUpward;
KeyState g_keyDownward;
KeyState g_keyTiltLeft;
KeyState g_keyTiltRight;
KeyState g_buttonRotate;
double g_mouseX;
double g_mouseY;

VLR::Point3D g_cameraPos;
VLR::Quaternion g_cameraOrientation;
float g_brightnessCoeff;
// PerspectiveCamera
float g_persSensitivity;
float g_fovYInDeg;
float g_lensRadius;
float g_objPlaneDistance;
// EquirectangularCamera
float g_equiSensitivity;
float g_phiAngle;
float g_thetaAngle;
int32_t g_cameraType;

template <typename T, size_t size>
constexpr size_t lengthof(const T(&array)[size]) {
    return size;
}

static std::string readTxtFile(const std::string& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
};

static void glfw_error_callback(int32_t error, const char* description) {
    debugPrintf("Error %d: %s\n", error, description);
}

struct Image2DCacheKey {
    std::string filepath;
    bool applyDegamma;

    bool operator<(const Image2DCacheKey &key) const {
        if (filepath < key.filepath)
            return true;
        else if (filepath == key.filepath)
            if (applyDegamma < key.applyDegamma)
                return true;
        return false;
    }
};

std::map<Image2DCacheKey, VLRCpp::Image2DRef> g_image2DCache;

static VLRCpp::Image2DRef loadImage2D(const VLRCpp::ContextRef &context, const std::string &filepath, bool applyDegamma) {
    using namespace VLRCpp;
    using namespace VLR;

    Image2DRef ret;

    Image2DCacheKey key{ filepath, applyDegamma };
    if (g_image2DCache.count(key))
        return g_image2DCache.at(key);

    bool fileExists = false;
    {
        std::ifstream ifs(filepath);
        fileExists = ifs.is_open();
    }
    if (!fileExists)
        return ret;

    debugPrintf("Read image: %s...", filepath.c_str());

    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
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
            curDataHead += width;
        }

        ret = context->createLinearImage2D(width, height, VLRDataFormat_RGBA16Fx4, applyDegamma, (uint8_t*)linearImageData);

        delete[] linearImageData;
    }
    else {
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load(filepath.c_str(), &width, &height, &n, 4);
        ret = context->createLinearImage2D(width, height, VLRDataFormat_RGBA8x4, applyDegamma, linearImageData);
        stbi_image_free(linearImageData);
    }

    debugPrintf("done.\n");

    g_image2DCache[key] = ret;

    return ret;
}

struct SurfaceAttributeTuple {
    VLRCpp::SurfaceMaterialRef material;
    VLRCpp::ShaderNodeSocket nodeNormal;
    VLRCpp::ShaderNodeSocket nodeAlpha;

    SurfaceAttributeTuple(const VLRCpp::SurfaceMaterialRef &_material, const VLRCpp::ShaderNodeSocket &_nodeNormal, const VLRCpp::ShaderNodeSocket &_nodeAlpha) :
        material(_material), nodeNormal(_nodeNormal), nodeAlpha(_nodeAlpha) {}
};

typedef SurfaceAttributeTuple(*CreateMaterialFunction)(const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &);
typedef VLRTangentType(*PerMeshFunction)(const aiMesh* mesh);

static SurfaceAttributeTuple createMaterialDefaultFunction(const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
    using namespace VLRCpp;
    using namespace VLR;

    aiReturn ret;
    (void)ret;
    aiString strValue;
    float color[3];

    aiMat->Get(AI_MATKEY_NAME, strValue);
    VLRDebugPrintf("Material: %s\n", strValue.C_Str());

    MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
    if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
        Image2DTextureShaderNodeRef tex = context->createImage2DTextureShaderNode();
        Image2DRef image = loadImage2D(context, pathPrefix + strValue.C_Str(), true);
        tex->setImage(image);
        mat->setNodeAlbedo(tex->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
    }
    else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
        RGBSpectrum spectrum(color[0], color[1], color[2]);
        mat->setImmediateValueAlbedo(spectrum);
    }
    else {
        RGBSpectrum spectrum(1.0f, 0.0f, 1.0f);
        mat->setImmediateValueAlbedo(spectrum);
    }

    return SurfaceAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
}

static VLRTangentType perMeshDefaultFunction(const aiMesh* mesh) {
    return VLRTangentType_VertexAttribute;
}

static void recursiveConstruct(const VLRCpp::ContextRef &context, const aiScene* objSrc, const aiNode* nodeSrc,
                               const std::vector<SurfaceAttributeTuple> &attrTuples, bool flipV, const PerMeshFunction &meshFunc,
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
            debugPrintf("ignored non triangle mesh.\n");
            continue;
        }

        auto surfMesh = context->createTriangleMeshSurfaceNode(mesh->mName.C_Str());
        const SurfaceAttributeTuple attrTuple = attrTuples[mesh->mMaterialIndex];
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

            Vertex outVtx{ Point3D(p.x, p.y, p.z), Normal3D(n.x, n.y, n.z), Vector3D(t.x, t.y, t.z), TexCoord2D(uv.x, flipV ? (1 - uv.y) : uv.y) };
            float dotNT = dot(outVtx.normal, outVtx.tangent);
            if (std::fabs(dotNT) >= 0.01f)
                outVtx.tangent = normalize(outVtx.tangent - dotNT * outVtx.normal);
            //SLRAssert(absDot(outVtx.normal, outVtx.tangent) < 0.01f, "shading normal and tangent must be orthogonal: %g", absDot(outVtx.normal, outVtx.tangent));
            vertices.push_back(outVtx);
        }
        surfMesh->setVertices(vertices.data(), vertices.size());

        VLRTangentType tangentType = meshFunc(mesh);

        meshIndices.clear();
        for (int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace &face = mesh->mFaces[f];
            meshIndices.push_back(face.mIndices[0]);
            meshIndices.push_back(face.mIndices[1]);
            meshIndices.push_back(face.mIndices[2]);
        }
        surfMesh->addMaterialGroup(meshIndices.data(), meshIndices.size(), surfMat, nodeNormal, nodeAlpha, tangentType);

        (*nodeOut)->addChild(surfMesh);
    }

    if (nodeSrc->mNumChildren) {
        for (int c = 0; c < nodeSrc->mNumChildren; ++c) {
            InternalNodeRef subNode;
            recursiveConstruct(context, objSrc, nodeSrc->mChildren[c], attrTuples, flipV, meshFunc, &subNode);
            if (subNode != nullptr)
                (*nodeOut)->addChild(subNode);
        }
    }
}

static void construct(const VLRCpp::ContextRef &context, const std::string &filePath, bool flipV, VLRCpp::InternalNodeRef* nodeOut, 
                      CreateMaterialFunction matFunc = createMaterialDefaultFunction, PerMeshFunction meshFunc = perMeshDefaultFunction) {
    using namespace VLRCpp;
    using namespace VLR;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath, aiProcess_Triangulate);
    if (!scene) {
        debugPrintf("Failed to load %s.\n", filePath.c_str());
        return;
    }
    debugPrintf("Reading: %s done.\n", filePath.c_str());

    std::string pathPrefix = filePath.substr(0, filePath.find_last_of("/") + 1);

    // create materials
    std::vector<SurfaceAttributeTuple> attrTuples;
    for (int m = 0; m < scene->mNumMaterials; ++m) {
        const aiMaterial* aiMat = scene->mMaterials[m];
        attrTuples.push_back(matFunc(context, aiMat, pathPrefix));
    }

    recursiveConstruct(context, scene, scene->mRootNode, attrTuples, flipV, meshFunc, nodeOut);

    debugPrintf("Constructing: %s done.\n", filePath.c_str());
}



struct Shot {
    VLRCpp::SceneRef scene;
    VLRCpp::CameraRef camera;
    VLRCpp::PerspectiveCameraRef perspectiveCamera;
    VLRCpp::EquirectangularCameraRef equirectangularCamera;
};
uint32_t g_renderTargetSizeX;
uint32_t g_renderTargetSizeY;
static void createCornellBoxScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(translate(0.0f, 0.0f, 0.0f)));

    TriangleMeshSurfaceNodeRef cornellBox = context->createTriangleMeshSurfaceNode("CornellBox");
    {
        std::vector<Vertex> vertices;

        // Floor
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 5.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f, -1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(5.0f, 5.0f) });
        vertices.push_back(Vertex{ Point3D(1.5f,  0.0f,  1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(5.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f,  1.5f), Normal3D(0,  1, 0), Vector3D(1,  0,  0), TexCoord2D(0.0f, 0.0f) });
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
            Image2DRef image = loadImage2D(context, "resources/checkerboard_line.png", true);
            Image2DTextureShaderNodeRef nodeAlbedo = context->createImage2DTextureShaderNode();
            nodeAlbedo->setImage(image);
            nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
            MatteSurfaceMaterialRef matMatte = context->createMatteSurfaceMaterial();
            matMatte->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }

        {
            MatteSurfaceMaterialRef matMatte = context->createMatteSurfaceMaterial();
            matMatte->setImmediateValueAlbedo(sRGB_degamma(VLR::RGBSpectrum(0.75f, 0.75f, 0.75f)));

            std::vector<uint32_t> matGroup = {
                4, 5, 6, 4, 6, 7,
                8, 9, 10, 8, 10, 11,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }

        {
            MatteSurfaceMaterialRef matMatte = context->createMatteSurfaceMaterial();
            matMatte->setImmediateValueAlbedo(sRGB_degamma(RGBSpectrum(0.75f, 0.25f, 0.25f)));

            //float value[3] = { 0.06f, 0.02f, 0.02f };
            //Float3TextureRef texEmittance = context->createConstantFloat3Texture(value);
            //SurfaceMaterialRef matMatte = context->createDiffuseEmitterSurfaceMaterial(texEmittance);

            std::vector<uint32_t> matGroup = {
                12, 13, 14, 12, 14, 15,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }

        {
            MatteSurfaceMaterialRef matMatte = context->createMatteSurfaceMaterial();
            matMatte->setImmediateValueAlbedo(sRGB_degamma(RGBSpectrum(0.25f, 0.25f, 0.75f)));

            std::vector<uint32_t> matGroup = {
                16, 17, 18, 16, 18, 19,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }

        {
            DiffuseEmitterSurfaceMaterialRef matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(RGBSpectrum(30.0f, 30.0f, 30.0f));

            std::vector<uint32_t> matGroup = {
                20, 21, 22, 20, 22, 23,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }

        {
            DiffuseEmitterSurfaceMaterialRef matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(RGBSpectrum(100.0f, 100.0f, 100.0f));

            std::vector<uint32_t> matGroup = {
                24, 25, 26, 24, 26, 27,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }
    }
    shot->scene->addChild(cornellBox);



    InternalNodeRef sphereNode;
    construct(context, "resources/sphere/sphere.obj", false, &sphereNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        ////// Aluminum
        ////RGBSpectrum eta(1.27579f, 0.940922f, 0.574879f);
        ////RGBSpectrum k(7.30257f, 6.33458f, 5.16694f);
        ////// Copper
        ////RGBSpectrum eta(0.237698f, 0.734847f, 1.37062f);
        ////RGBSpectrum k(3.44233f, 2.55751f, 2.23429f);
        ////// Gold
        ////RGBSpectrum eta(0.12481f, 0.468228f, 1.44476f);
        ////RGBSpectrum k(3.32107f, 2.23761f, 1.69196f);
        ////// Iron
        ////RGBSpectrum eta(2.91705f, 2.92092f, 2.53253f);
        ////RGBSpectrum k(3.06696f, 2.93804f, 2.7429f);
        ////// Lead
        ////RGBSpectrum eta(1.9566f, 1.82777f, 1.46089f);
        ////RGBSpectrum k(3.49593f, 3.38158f, 3.17737f);
        ////// Mercury
        ////RGBSpectrum eta(1.99144f, 1.5186f, 1.00058f);
        ////RGBSpectrum k(5.25161f, 4.6095f, 3.7646f);
        ////// Platinum
        ////RGBSpectrum eta(2.32528f, 2.06722f, 1.81479f);
        ////RGBSpectrum k(4.19238f, 3.67941f, 3.06551f);
        //// Silver
        //RGBSpectrum eta(0.157099f, 0.144013f, 0.134847f);
        //RGBSpectrum k(3.82431f, 3.1451f, 2.27711f);
        ////// Titanium
        ////RGBSpectrum eta(2.71866f, 2.50954f, 2.22767f);
        ////RGBSpectrum k(3.79521f, 3.40035f, 3.00114f);
        //SpecularReflectionSurfaceMaterialRef mat = context->createSpecularReflectionSurfaceMaterial();
        //mat->setImmediateValueCoeffR(RGBSpectrum(0.999f, 0.999f, 0.999f));
        //mat->setImmediateValueEta(eta);
        //mat->setImmediateValue_k(k);

        // Air
        RGBSpectrum etaExt(1.00036f, 1.00021f, 1.00071f);
        //// Water
        //RGBSpectrum etaInt(1.33161f, 1.33331f, 1.33799f);
        //// Glass BK7
        //RGBSpectrum etaInt(1.51455f, 1.51816f, 1.52642f);
        // Diamond
        RGBSpectrum etaInt(2.41174f, 2.42343f, 2.44936f);
        SpecularScatteringSurfaceMaterialRef mat = context->createSpecularScatteringSurfaceMaterial();
        mat->setImmediateValueCoeff(RGBSpectrum(0.999f, 0.999f, 0.999f));
        mat->setImmediateValueEtaExt(etaExt);
        mat->setImmediateValueEtaInt(etaInt);

        //float coeff[] = { 0.5f, 0.5f, 0.5f };
        //// Silver
        //float eta[] = { 0.157099f, 0.144013f, 0.134847f };
        //float k[] = { 3.82431f, 3.1451f, 2.27711f };
        //Float3TextureRef texCoeff = context->createConstantFloat3Texture(coeff);
        //Float3TextureRef texEta = context->createConstantFloat3Texture(eta);
        //Float3TextureRef tex_k = context->createConstantFloat3Texture(k);
        //SurfaceMaterialRef matA = context->createSpecularReflectionSurfaceMaterial(texCoeff, texEta, tex_k);

        //float albedoRoughness[] = { 0.75f, 0.25f, 0.0f, 0.0f };
        //Float4TextureRef texAlbedoRoughness = context->createConstantFloat4Texture(albedoRoughness);
        //SurfaceMaterialRef matB = context->createMatteSurfaceMaterial(texAlbedoRoughness);

        //SurfaceMaterialRef mats[] = { matA, matB };
        //SurfaceMaterialRef mat = context->createMultiSurfaceMaterial(mats, lengthof(mats));

        return SurfaceAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
    });
    shot->scene->addChild(sphereNode);
    sphereNode->setTransform(context->createStaticTransform(scale(0.5f) * translate<float>(0.0f, 1.0f, 0.0f)));



    //Image2DRef imgEnv = loadImage2D(context, "resources/environments/WhiteOne.exr");
    //Float3TextureRef texEnv = context->createImageFloat3Texture(imgEnv);
    //EnvironmentEmitterSurfaceMaterialRef matEnv = context->createEnvironmentEmitterSurfaceMaterial(texEnv);
    //scene->setEnvironment(matEnv);

    g_cameraPos = Point3D(0, 1.5f, 6.0f);
    g_cameraOrientation = qRotateY<float>(M_PI);
    g_brightnessCoeff = 1.0f;

    g_renderTargetSizeX = 1280;
    g_renderTargetSizeY = 720;

    g_persSensitivity = 1.0f;
    g_fovYInDeg = 40;
    g_lensRadius = 0.0f;
    g_objPlaneDistance = 1.0f;
    shot->perspectiveCamera = context->createPerspectiveCamera(g_cameraPos, g_cameraOrientation,
                                                               g_persSensitivity, (float)g_renderTargetSizeX / g_renderTargetSizeY, g_fovYInDeg * M_PI / 180, g_lensRadius, 1.0f, g_objPlaneDistance);

    g_equiSensitivity = 1.0f / (g_phiAngle * (1 - std::cos(g_thetaAngle)));
    g_phiAngle = M_PI;
    g_thetaAngle = g_phiAngle * g_renderTargetSizeY / g_renderTargetSizeX;
    shot->equirectangularCamera = context->createEquirectangularCamera(g_cameraPos, g_cameraOrientation,
                                                                       g_equiSensitivity, g_phiAngle, g_thetaAngle);

    g_cameraType = 0;
    shot->camera = shot->perspectiveCamera;
}
static void createMaterialTestScene(const VLRCpp::ContextRef &context, Shot* shot) {
    using namespace VLRCpp;
    using namespace VLR;

    shot->scene = context->createScene(context->createStaticTransform(rotateY<float>(M_PI / 2) * translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", true, &modelNode, [](const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace VLRCpp;
        using namespace VLR;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        OffsetAndScaleUVTextureMap2DShaderNodeRef nodeTexCoord = context->createOffsetAndScaleUVTextureMap2DShaderNode();
        nodeTexCoord->setValues(offset, scale);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", true);

        Image2DTextureShaderNodeRef nodeAlbedo = context->createImage2DTextureShaderNode();
        nodeAlbedo->setImage(image);
        nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest, VLRTextureFilter_None);
        nodeAlbedo->setNodeTexCoord(nodeTexCoord->getSocket(VLRShaderNodeSocketType_TextureCoordinates, 0));

        MatteSurfaceMaterialRef mat = context->createMatteSurfaceMaterial();
        mat->setNodeAlbedo(nodeAlbedo->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

        return SurfaceAttributeTuple(mat, ShaderNodeSocket(), ShaderNodeSocket());
    });
    shot->scene->addChild(modelNode);



    TriangleMeshSurfaceNodeRef light = context->createTriangleMeshSurfaceNode("light");
    {
        std::vector<Vertex> vertices;

        // Light
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, 0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(0.5f, 0.0f, -0.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

        light->setVertices(vertices.data(), vertices.size());

        {
            DiffuseEmitterSurfaceMaterialRef matLight = context->createDiffuseEmitterSurfaceMaterial();
            matLight->setImmediateValueEmittance(RGBSpectrum(30.0f, 30.0f, 30.0f));

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodeSocket(), ShaderNodeSocket(), VLRTangentType_VertexAttribute);
        }
    }
    InternalNodeRef lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, -3.0f) * rotateX<float>(M_PI / 2)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, "resources/material_test/mitsuba_knob.obj", false, &modelNode, 
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
            matteMat->setImmediateValueAlbedo(RGBSpectrum(0.18f, 0.18f, 0.18f));

            mat = matteMat;
        }
        else if (strcmp(strValue.C_Str(), "Glossy") == 0) {
            Image2DRef imgNormalAlpha = loadImage2D(context, pathPrefix + "TexturesCom_Leaves0165_1_alphamasked_S.png", true);
            Image2DTextureShaderNodeRef nodeBaseColorAlpha = context->createImage2DTextureShaderNode();
            nodeBaseColorAlpha->setImage(imgNormalAlpha);

            UE4SurfaceMaterialRef ue4Mat = context->createUE4SurfaceMaterial();
            ue4Mat->setNodeBaseColor(nodeBaseColorAlpha->getSocket(VLRShaderNodeSocketType_Spectrum, 0));
            ue4Mat->setImmediateValueBaseColor(sRGB_degamma(RGBSpectrum(0.75f, 0.5f, 0.0025f)));
            ue4Mat->setImmediateValueOcclusion(0.0f);
            ue4Mat->setImmediateValueRoughness(0.3f);
            ue4Mat->setImmediateValueMetallic(0.0f);

            mat = ue4Mat;

            socketAlpha = nodeBaseColorAlpha->getSocket(VLRShaderNodeSocketType_float, 3);

            //MicrofacetReflectionSurfaceMaterialRef mfMat = context->createMicrofacetReflectionSurfaceMaterial();
            //// Aluminum
            //mfMat->setImmediateValueEta(RGBSpectrum(1.27579f, 0.940922f, 0.574879f));
            //mfMat->setImmediateValue_k(RGBSpectrum(7.30257f, 6.33458f, 5.16694f));
            //float roughness[] = { 0.1f, 0.3f };
            //mfMat->setImmediateValueRoughness(roughness);

            //mat = mfMat;

            //GeometryShaderNodeRef nodeGeom = context->createGeometryShaderNode();
            //Vector3DToSpectrumShaderNodeRef nodeVec2Sp = context->createVector3DToSpectrumShaderNode();
            //nodeVec2Sp->setNodeVector3D(nodeGeom->getSocket(VLRShaderNodeSocketType_Vector3D, 0));
            //MatteSurfaceMaterialRef mtMat = context->createMatteSurfaceMaterial();
            //mtMat->setNodeAlbedo(nodeVec2Sp->getSocket(VLRShaderNodeSocketType_Spectrum, 0));

            //mat = mtMat;
        }
        //if (strcmp(strValue.C_Str(), "Base") == 0 || 
        //    strcmp(strValue.C_Str(), "Glossy") == 0) {
        //    float coeff[] = { 0.99f, 0.99f, 0.99f };
        //    float etaExt[] = { 1.0f, 1.0f, 1.0f };
        //    float etaInt[] = { 1.4f, 1.4f, 1.4f };
        //    float roughness[] = { 0.1f, 0.1f };
        //    Float3TextureRef texCoeff = context->createConstantFloat3Texture(coeff);
        //    Float3TextureRef texEtaExt = context->createConstantFloat3Texture(etaExt);
        //    Float3TextureRef texEtaInt = context->createConstantFloat3Texture(etaInt);
        //    Float2TextureRef texRoughness = context->createConstantFloat2Texture(roughness);
        //    mat = context->createMicrofacetScatteringSurfaceMaterial(texCoeff, texEtaExt, texEtaInt, texRoughness, nullptr);

        //    //float coeff[] = { 0.99f, 0.99f, 0.99f };
        //    //float F0 = 0.2f;
        //    //Float3TextureRef texCoeff = context->createConstantFloat3Texture(coeff);
        //    //FloatTextureRef texF0 = context->createConstantFloatTexture(F0);
        //    //mat = context->createLambertianScatteringSurfaceMaterial(texCoeff, texF0, nullptr);
        //}

        return SurfaceAttributeTuple(mat, socketNormal, socketAlpha);
    },
              [](const aiMesh* mesh) {
        return VLRTangentType_RadialY;
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.04089, 0)));



    //construct(context, "../../assets/spman/spman.obj", true, &modelNode, [](VLRCpp::Context &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
    //    using namespace VLRCpp;
    //    using namespace VLR;

    //    aiReturn ret;
    //    (void)ret;
    //    aiString strValue;
    //    float color[3];

    //    aiMat->Get(AI_MATKEY_NAME, strValue);

    //    SurfaceMaterialRef mat;
    //    Float4TextureRef texNormalAlpha;
    //    if (strcmp(strValue.C_Str(), "_Head1") == 0) {
    //        Float3TextureRef texBaseColor = context->createImageFloat3Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_BaseColor.png", true));
    //        Float3TextureRef texOcclusionRoughnessMetallic = context->createImageFloat3Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_OcclusionRoughnessMetallic.png", false));
    //        texNormalAlpha = context->createImageFloat4Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_NormalAlpha.png", false));
    //        mat = context->createUE4SurfaceMaterial(texBaseColor, texOcclusionRoughnessMetallic, nullptr);
    //    }
    //    else if (strcmp(strValue.C_Str(), "_Body1") == 0) {
    //        Float3TextureRef texBaseColor = context->createImageFloat3Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_BaseColor.png", true));
    //        Float3TextureRef texOcclusionRoughnessMetallic = context->createImageFloat3Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_OcclusionRoughnessMetallic.png", false));
    //        texNormalAlpha = context->createImageFloat4Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_NormalAlpha.png", false));
    //        mat = context->createUE4SurfaceMaterial(texBaseColor, texOcclusionRoughnessMetallic, nullptr);
    //    }
    //    else if (strcmp(strValue.C_Str(), "_Base1") == 0) {
    //        Float3TextureRef texBaseColor = context->createImageFloat3Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_BaseColor.png", true));
    //        Float3TextureRef texOcclusionRoughnessMetallic = context->createImageFloat3Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_OcclusionRoughnessMetallic.png", false));
    //        texNormalAlpha = context->createImageFloat4Texture(
    //            loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_NormalAlpha.png", false));
    //        mat = context->createUE4SurfaceMaterial(texBaseColor, texOcclusionRoughnessMetallic, nullptr);
    //    }

    //    return SurfaceAttributeTuple(mat, texNormalAlpha);
    //});
    //shot->scene->addChild(modelNode);
    //modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.01, 0) * scale<float>(0.25f)));



    Image2DRef imgEnv = loadImage2D(context, "resources/material_test/Chelsea_Stairs_3k.exr", false);
    EnvironmentTextureShaderNodeRef nodeEnvTex = context->createEnvironmentTextureShaderNode();
    nodeEnvTex->setImage(imgEnv);
    EnvironmentEmitterSurfaceMaterialRef matEnv = context->createEnvironmentEmitterSurfaceMaterial();
    matEnv->setNodeEmittance(nodeEnvTex);
    //matEnv->setImmediateValueEmittance(RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->scene->setEnvironment(matEnv);



    g_cameraPos = Point3D(10.0f, 5.0f, 0.0f);
    g_cameraOrientation = qRotateY<float>(-M_PI / 2) * qRotateX<float>(18 * M_PI / 180);
    g_brightnessCoeff = 1.0f;

    g_renderTargetSizeX = 1280;
    g_renderTargetSizeY = 720;

    g_persSensitivity = 1.0f;
    g_fovYInDeg = 40;
    g_lensRadius = 0.0f;
    g_objPlaneDistance = 1.0f;
    shot->perspectiveCamera = context->createPerspectiveCamera(g_cameraPos, g_cameraOrientation,
                                                               g_persSensitivity, (float)g_renderTargetSizeX / g_renderTargetSizeY, g_fovYInDeg * M_PI / 180, g_lensRadius, 1.0f, g_objPlaneDistance);

    g_equiSensitivity = 1.0f / (g_phiAngle * (1 - std::cos(g_thetaAngle)));
    g_phiAngle = M_PI;
    g_thetaAngle = g_phiAngle * g_renderTargetSizeY / g_renderTargetSizeX;
    shot->equirectangularCamera = context->createEquirectangularCamera(g_cameraPos, g_cameraOrientation,
                                                                       g_equiSensitivity, g_phiAngle, g_thetaAngle);

    g_cameraType = 0;
    shot->camera = shot->perspectiveCamera;
}



static int32_t mainFunc(int32_t argc, const char* argv[]) {
    StopWatch swGlobal;

    swGlobal.start();

    using namespace VLRCpp;
    using namespace VLR;

    std::set<int32_t> devices;
    bool enableLogging = false;
    bool enableGUI = true;
    uint32_t renderImageSizeX = 1920;
    uint32_t renderImageSizeY = 1080;
    uint32_t stackSize = 0;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--", 2) == 0) {
            if (strcmp(argv[i] + 2, "list") == 0) {
                vlrPrintDevices();
            }
            else if (strcmp(argv[i] + 2, "devices") == 0) {
                ++i;
                for (; i < argc; ++i) {
                    if (strncmp(argv[i], "--", 2) == 0) {
                        break;
                    }
                    devices.insert(atoi(argv[i]));
                }
                --i;
            }
            else if (strcmp(argv[i] + 2, "logging") == 0) {
                enableLogging = true;
            }
            else if (strcmp(argv[i] + 2, "nodisplay") == 0) {
                enableGUI = false;
            }
            else if (strcmp(argv[i] + 2, "imagesize") == 0) {
                ++i;
                renderImageSizeX = atoi(argv[i]);
                ++i;
                renderImageSizeY = atoi(argv[i]);
            }
            else if (strcmp(argv[i] + 2, "stacksize") == 0) {
                ++i;
                if (strncmp(argv[i], "--", 2) != 0)
                    stackSize = atoi(argv[i]);
            }
        }
    }

    VLRCpp::ContextRef context = VLRCpp::Context::create(enableLogging, stackSize);

    int32_t primaryDevice = 0;
    if (!devices.empty()) {
        std::vector<int32_t> deviceArray;
        for (auto it = devices.cbegin(); it != devices.cend(); ++it)
            deviceArray.push_back(*it);
        context->setDevices(deviceArray.data(), deviceArray.size());

        primaryDevice = deviceArray.front();
    }

    char deviceName[128];
    vlrGetDeviceName(primaryDevice, deviceName, lengthof(deviceName));

    Shot shot;
    //createCornellBoxScene(context, &shot);
    createMaterialTestScene(context, &shot);



    if (enableGUI) {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return 1;

        GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();

        // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
        const uint32_t OpenGLMajorVersion = 4;
        const uint32_t OpenGLMinorVersion = 6;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(Platform_macOS)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        // JP: ウインドウの初期化。
        //     HiDPIディスプレイに対応する。
        float contentScaleX, contentScaleY;
        glfwGetMonitorContentScale(primaryMonitor, &contentScaleX, &contentScaleY);
        const float UIScaling = contentScaleX;
        GLFWwindow* window = glfwCreateWindow((int32_t)(g_renderTargetSizeX * UIScaling), (int32_t)(g_renderTargetSizeY * UIScaling), "VLR", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return -1;
        }

        int32_t curFBWidth;
        int32_t curFBHeight;
        glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

        glfwMakeContextCurrent(window);

        glfwSwapInterval(1); // Enable vsync

        // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
        int32_t gl3wRet = gl3wInit();
        if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
            glfwTerminate();
            debugPrintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
            return -1;
        }



        // Setup ImGui binding
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
        ImGui_ImplGlfwGL3_Init(window, true);

        // Setup style
        ImGui::StyleColorsDark();



        // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
        GLTK::VertexArray vertexArrayForFullScreen;
        vertexArrayForFullScreen.initialize();

        GLTK::Buffer outputBufferGL;
        outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(VLR::RGBSpectrum), g_renderTargetSizeX * g_renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);

        context->bindOutputBuffer(g_renderTargetSizeX, g_renderTargetSizeY, outputBufferGL.getRawHandle());

        GLTK::BufferTexture outputTexture;
        outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGB32F);

        // JP: OptiXの出力を書き出すシェーダー。
        GLTK::GraphicsShader drawOptiXResultShader;
        drawOptiXResultShader.initializeVSPS(readTxtFile("resources/shaders/drawOptiXResult.vert"),
                                             readTxtFile("resources/shaders/drawOptiXResult.frag"));

        // JP: HiDPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
        GLTK::FrameBuffer frameBuffer;
        frameBuffer.initialize(g_renderTargetSizeX, g_renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

        // JP: アップスケール用のシェーダー。
        GLTK::GraphicsShader scaleShader;
        scaleShader.initializeVSPS(readTxtFile("resources/shaders/scale.vert"),
                                   readTxtFile("resources/shaders/scale.frag"));

        // JP: アップスケール用のサンプラー。
        //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
        GLTK::Sampler scaleSampler;
        scaleSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest, GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



        glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

            switch (button) {
            case GLFW_MOUSE_BUTTON_MIDDLE: {
                debugPrintf("Mouse Middle\n");
                g_buttonRotate.recordStateChange(action == GLFW_PRESS, g_frameIndex);
                break;
            }
            default:
                break;
            }
        });
        glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
            g_mouseX = x;
            g_mouseY = y;
        });
        glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

            switch (key) {
            case GLFW_KEY_W: {
                debugPrintf("W: %d\n", action);
                g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_S: {
                debugPrintf("S: %d\n", action);
                g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_A: {
                debugPrintf("A: %d\n", action);
                g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_D: {
                debugPrintf("D: %d\n", action);
                g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_R: {
                debugPrintf("R: %d\n", action);
                g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_F: {
                debugPrintf("F: %d\n", action);
                g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_Q: {
                debugPrintf("Q: %d\n", action);
                g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_E: {
                debugPrintf("E: %d\n", action);
                g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            default:
                break;
            }
        });

        StopWatch sw;
        uint64_t accumFrameTimes = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            bool operatingCamera = false;
            bool cameraIsActuallyMoving = false;

            bool resized = false;
            int32_t newFBWidth;
            int32_t newFBHeight;
            glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
            if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
                curFBWidth = newFBWidth;
                curFBHeight = newFBHeight;

                g_renderTargetSizeX = curFBWidth / UIScaling;
                g_renderTargetSizeY = curFBHeight / UIScaling;

                frameBuffer.finalize();
                outputTexture.finalize();
                outputBufferGL.finalize();

                outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(VLR::RGBSpectrum), g_renderTargetSizeX * g_renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);

                context->bindOutputBuffer(g_renderTargetSizeX, g_renderTargetSizeY, outputBufferGL.getRawHandle());

                outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGB32F);

                frameBuffer.initialize(g_renderTargetSizeX, g_renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

                shot.perspectiveCamera = context->createPerspectiveCamera(g_cameraPos, g_cameraOrientation,
                                                                          g_persSensitivity, (float)g_renderTargetSizeX / g_renderTargetSizeY, g_fovYInDeg * M_PI / 180, g_lensRadius, 1.0f, g_objPlaneDistance);

                resized = true;
            }

            // process key events
            Quaternion tempOrientation;
            {
                int32_t trackZ = 0;
                if (g_keyForward.getState() == true) {
                    if (g_keyBackward.getState() == true)
                        trackZ = 0;
                    else
                        trackZ = 1;
                }
                else {
                    if (g_keyBackward.getState() == true)
                        trackZ = -1;
                    else
                        trackZ = 0;
                }

                int32_t trackX = 0;
                if (g_keyLeftward.getState() == true) {
                    if (g_keyRightward.getState() == true)
                        trackX = 0;
                    else
                        trackX = 1;
                }
                else {
                    if (g_keyRightward.getState() == true)
                        trackX = -1;
                    else
                        trackX = 0;
                }

                int32_t trackY = 0;
                if (g_keyUpward.getState() == true) {
                    if (g_keyDownward.getState() == true)
                        trackY = 0;
                    else
                        trackY = 1;
                }
                else {
                    if (g_keyDownward.getState() == true)
                        trackY = -1;
                    else
                        trackY = 0;
                }

                int32_t tiltZ = 0;
                if (g_keyTiltRight.getState() == true) {
                    if (g_keyTiltLeft.getState() == true)
                        tiltZ = 0;
                    else
                        tiltZ = 1;
                }
                else {
                    if (g_keyTiltLeft.getState() == true)
                        tiltZ = -1;
                    else
                        tiltZ = 0;
                }

                static double deltaX = 0, deltaY = 0;
                static double lastX, lastY;
                static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
                if (g_buttonRotate.getState() == true) {
                    if (g_buttonRotate.getTime() == g_frameIndex) {
                        lastX = g_mouseX;
                        lastY = g_mouseY;
                    }
                    else {
                        deltaX = g_mouseX - lastX;
                        deltaY = g_mouseY - lastY;
                    }
                }

                float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
                Vector3D axis(deltaY, -deltaX, 0);
                axis /= deltaAngle;
                if (deltaAngle == 0.0f)
                    axis = Vector3D(1, 0, 0);

                g_cameraOrientation = g_cameraOrientation * qRotateZ(0.025f * tiltZ);
                tempOrientation = g_cameraOrientation * qRotate(0.15f * 1e-2f * deltaAngle, axis);
                g_cameraPos += tempOrientation.toMatrix3x3() * 0.05f * Vector3D(trackX, trackY, trackZ);
                if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == g_frameIndex) {
                    g_cameraOrientation = tempOrientation;
                    deltaX = 0;
                    deltaY = 0;
                }

                operatingCamera = (g_keyForward.getState() || g_keyBackward.getState() ||
                                   g_keyLeftward.getState() || g_keyRightward.getState() ||
                                   g_keyUpward.getState() || g_keyDownward.getState() ||
                                   g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                                   g_buttonRotate.getState());
                cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 || tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY)) && operatingCamera;

                g_prevMouseX = g_mouseX;
                g_prevMouseY = g_mouseY;
            }

            {
                ImGui_ImplGlfwGL3_NewFrame(g_renderTargetSizeX, g_renderTargetSizeY, UIScaling);

                {
                    ImGui::Begin("Misc", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                    ImGui::Text("Device: %s", deviceName);

                    ImGui::End();
                }

                bool cameraSettingsChanged = resized;
                static bool g_forceLowResolution = false;
                static uint32_t g_numAccumFrames = 1;
                {
                    ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                    cameraSettingsChanged |= ImGui::InputFloat3("Position", (float*)&g_cameraPos);
                    ImGui::SliderFloat("Brightness", &g_brightnessCoeff, 0.01f, 10.0f, "%.3f", 2.0f);
                    cameraSettingsChanged |= ImGui::Checkbox("Force Low Resolution", &g_forceLowResolution);

                    const char* CameraTypeNames[] = { "Perspective", "Equirectangular" };
                    cameraSettingsChanged |= ImGui::Combo("Camera Type", &g_cameraType, CameraTypeNames, lengthof(CameraTypeNames));

                    if (g_cameraType == 0) {
                        cameraSettingsChanged |= ImGui::SliderFloat("fov Y", &g_fovYInDeg, 1, 179, "%.3f", 1.0f);
                        cameraSettingsChanged |= ImGui::SliderFloat("Lens Radius", &g_lensRadius, 0.0f, 0.15f, "%.3f", 1.0f);
                        cameraSettingsChanged |= ImGui::SliderFloat("Object Plane Distance", &g_objPlaneDistance, 0.01f, 20.0f, "%.3f", 2.0f);

                        g_persSensitivity = g_lensRadius == 0.0f ? 1.0f : 1.0f / (M_PI * g_lensRadius * g_lensRadius);

                        shot.camera = shot.perspectiveCamera;
                    }
                    else if (g_cameraType == 1) {
                        cameraSettingsChanged |= ImGui::SliderFloat("Phi Angle", &g_phiAngle, M_PI / 18, 2 * M_PI);
                        cameraSettingsChanged |= ImGui::SliderFloat("Theta Angle", &g_thetaAngle, M_PI / 18, 1 * M_PI);

                        g_equiSensitivity = 1.0f / (g_phiAngle * (1 - std::cos(g_thetaAngle)));

                        shot.camera = shot.equirectangularCamera;
                    }

                    ImGui::Text("%u [spp], %g [ms/sample]", g_numAccumFrames, (float)accumFrameTimes / (g_numAccumFrames - 1));

                    ImGui::End();
                }

                {
                    ImGui::Begin("Scene");

                    ImGui::BeginChild("Hierarchy", ImVec2(-1, 300), false);

                    struct SelectedChild {
                        InternalNodeRef parent;
                        int32_t childIndex;

                        bool operator<(const SelectedChild &v) const {
                            if (parent < v.parent) {
                                return true;
                            }
                            else if (parent == v.parent) {
                                if (childIndex < v.childIndex)
                                    return true;
                            }
                            return false;
                        }
                    };

                    static std::set<SelectedChild> g_selectedNodes;

                    const std::function<SelectedChild(InternalNodeRef)> recursiveBuild = [&recursiveBuild](InternalNodeRef parent) {
                        SelectedChild clickedChild{ nullptr, -1 };

                        for (int i = 0; i < parent->getNumChildren(); ++i) {
                            NodeRef child = parent->getChildAt(i);
                            SelectedChild curChild{ parent, i };

                            ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                            if (g_selectedNodes.count(curChild))
                                node_flags |= ImGuiTreeNodeFlags_Selected;
                            if (child->getNodeType() == NodeType::InternalNode) {
                                bool nodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                                bool mouseOnLabel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) > ImGui::GetTreeNodeToLabelSpacing();
                                if (ImGui::IsItemClicked() && mouseOnLabel)
                                    clickedChild = curChild;
                                if (nodeOpen) {
                                    SelectedChild cSelectedChild = recursiveBuild(std::dynamic_pointer_cast<InternalNodeHolder>(child));
                                    if (cSelectedChild.childIndex != -1)
                                        clickedChild = cSelectedChild;
                                }
                            }
                            else {
                                node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
                                ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                                if (ImGui::IsItemClicked())
                                    clickedChild = curChild;
                            }
                        }

                        ImGui::TreePop();

                        return clickedChild;
                    };

                    SelectedChild clickedChild{ nullptr, -1 };

                    for (int i = 0; i < shot.scene->getNumChildren(); ++i) {
                        NodeRef child = shot.scene->getChildAt(i);
                        SelectedChild curChild{ nullptr, i };

                        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                        if (g_selectedNodes.count(curChild))
                            node_flags |= ImGuiTreeNodeFlags_Selected;
                        if (child->getNodeType() == NodeType::InternalNode) {
                            bool nodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                            bool mouseOnLabel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) > ImGui::GetTreeNodeToLabelSpacing();
                            if (ImGui::IsItemClicked() && mouseOnLabel)
                                clickedChild = curChild;
                            if (nodeOpen) {
                                SelectedChild cSelectedChild = recursiveBuild(std::dynamic_pointer_cast<InternalNodeHolder>(child));
                                if (cSelectedChild.childIndex != -1)
                                    clickedChild = cSelectedChild;
                            }
                        }
                        else {
                            node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
                            ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                            if (ImGui::IsItemClicked())
                                clickedChild = curChild;
                        }
                    }

                    // JP: 何かクリックした要素がある場合。
                    bool newOnlyOneSelected = false;
                    if (clickedChild.childIndex != -1) {
                        if (ImGui::GetIO().KeyCtrl) {
                            // JP: Ctrlキーを押しながら選択した場合は追加選択or選択解除。
                            if (g_selectedNodes.count(clickedChild))
                                g_selectedNodes.erase(clickedChild);
                            else
                                g_selectedNodes.insert(clickedChild);
                        }
                        else {
                            if (g_selectedNodes.count(clickedChild)) {
                                // JP: クリックした要素を既に選択リストに持っていた場合は全ての選択状態を解除する。
                                //     このとき他に選択要素を持っていた場合はクリックした要素だけを選択状態にする。
                                bool multiplySelected = g_selectedNodes.size() > 1;
                                g_selectedNodes.clear();
                                if (multiplySelected)
                                    g_selectedNodes.insert(clickedChild);
                            }
                            else {
                                // JP: 全ての選択状態を解除してクリックした要素だけを選択状態にする。
                                g_selectedNodes.clear();
                                g_selectedNodes.insert(clickedChild);
                            }
                        }

                        // JP: クリック時には必ず選択状態に何らかの変化が起きるので、
                        //     クリック後に選択要素数が1であれば、必ずそれは新たにひとつだけ選択された要素となる。
                        if (g_selectedNodes.size() == 1)
                            newOnlyOneSelected = true;
                    }

                    ImGui::EndChild();

                    ImGui::Separator();

                    NodeRef node;

                    if (g_selectedNodes.size() == 1) {
                        const SelectedChild &sc = *g_selectedNodes.cbegin();
                        if (sc.parent)
                            node = sc.parent->getChildAt(sc.childIndex);
                        else
                            node = shot.scene->getChildAt(sc.childIndex);
                    }

                    static char g_nodeName[256];
                    if (newOnlyOneSelected) {
                        size_t copySize = std::min(std::strlen(node->getName()), sizeof(g_nodeName) - 1);
                        std::memcpy(g_nodeName, node->getName(), copySize);
                        g_nodeName[copySize] = '\0';
                    }
                    else if (g_selectedNodes.size() != 1) {
                        g_nodeName[0] = '\0';
                    }

                    if (node) {
                        ImGui::AlignTextToFramePadding();
                        ImGui::Text("Name:"); ImGui::SameLine();
                        ImGui::PushID("NameTextBox");
                        if (ImGui::InputText("", g_nodeName, sizeof(g_nodeName), ImGuiInputTextFlags_EnterReturnsTrue)) {
                            node->setName(g_nodeName);
                        }
                        ImGui::PopID();

                        if (node->getNodeType() == NodeType::InternalNode) {

                        }
                        else {

                        }
                    }

                    ImGui::End();
                }

                if (g_cameraType == 0) {
                    shot.perspectiveCamera->setPosition(g_cameraPos);
                    shot.perspectiveCamera->setOrientation(tempOrientation);
                    if (cameraSettingsChanged) {
                        shot.perspectiveCamera->setSensitivity(g_persSensitivity);
                        shot.perspectiveCamera->setFovY(g_fovYInDeg * M_PI / 180);
                        shot.perspectiveCamera->setLensRadius(g_lensRadius);
                        shot.perspectiveCamera->setObjectPlaneDistance(g_objPlaneDistance);
                    }
                }
                else if (g_cameraType == 1) {
                    shot.equirectangularCamera->setPosition(g_cameraPos);
                    shot.equirectangularCamera->setOrientation(tempOrientation);
                    if (cameraSettingsChanged) {
                        shot.equirectangularCamera->setSensitivity(g_equiSensitivity);
                        shot.equirectangularCamera->setAngles(g_phiAngle, g_thetaAngle);
                    }
                }

                static bool g_operatedCameraOnPrevFrame = false;
                uint32_t shrinkCoeff = (operatingCamera || g_forceLowResolution) ? 4 : 1;

                bool firstFrame = cameraIsActuallyMoving || (g_operatedCameraOnPrevFrame ^ operatingCamera) || cameraSettingsChanged;
                if (firstFrame)
                    accumFrameTimes = 0;
                else
                    sw.start();
                context->render(shot.scene, shot.camera, shrinkCoeff, firstFrame, &g_numAccumFrames);
                if (!firstFrame)
                    accumFrameTimes += sw.stop(StopWatch::Milliseconds);

                //// DELETE ME
                //if (g_numAccumFrames == 32) {
                //    debugPrintf("Camera:\n");
                //    debugPrintf("Position: %g, %g, %g\n", g_cameraPos.x, g_cameraPos.y, g_cameraPos.z);
                //    debugPrintf("Orientation: %g, %g, %g, %g\n", g_cameraOrientation.x, g_cameraOrientation.y, g_cameraOrientation.z, g_cameraOrientation.w);

                //    auto output = (const RGBSpectrum*)context->mapOutputBuffer();
                //    auto data = new uint32_t[renderTargetSizeX * renderTargetSizeY];
                //    for (int y = 0; y < renderTargetSizeY; ++y) {
                //        for (int x = 0; x < renderTargetSizeX; ++x) {
                //            RGBSpectrum srcPix = output[y * renderTargetSizeX + x];
                //            uint32_t &pix = data[y * renderTargetSizeX + x];

                //            srcPix *= g_brightnessCoeff;
                //            srcPix = RGBSpectrum::One() - exp(-srcPix);
                //            srcPix = sRGB_gamma(srcPix);

                //            pix = ((std::min<uint8_t>(srcPix.r * 256, 255) << 0) |
                //                   (std::min<uint8_t>(srcPix.g * 256, 255) << 8) |
                //                   (std::min<uint8_t>(srcPix.b * 256, 255) << 16) |
                //                   (0xFF << 24));
                //        }
                //    }
                //    stbi_write_png("output.png", renderTargetSizeX, renderTargetSizeY, 4, data, sizeof(data[0]) * renderTargetSizeX);
                //    delete[] data;
                //    context->unmapOutputBuffer();
                //}

                g_operatedCameraOnPrevFrame = operatingCamera;

                // ----------------------------------------------------------------
                // JP: OptiXの出力とImGuiの描画。

                frameBuffer.bind(GLTK::FrameBuffer::Target::ReadDraw);

                glViewport(0, 0, frameBuffer.getWidth(), frameBuffer.getHeight());
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClearDepth(1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                {
                    drawOptiXResultShader.useProgram();

                    glUniform1i(0, (int32_t)g_renderTargetSizeX); GLTK::errorCheck();

                    glUniform1f(1, (float)shrinkCoeff); GLTK::errorCheck();

                    glUniform1f(2, g_brightnessCoeff); GLTK::errorCheck();

                    glActiveTexture(GL_TEXTURE0); GLTK::errorCheck();
                    outputTexture.bind();

                    vertexArrayForFullScreen.bind();
                    glDrawArrays(GL_TRIANGLES, 0, 3); GLTK::errorCheck();
                    vertexArrayForFullScreen.unbind();

                    outputTexture.unbind();
                }

                ImGui::Render();
                ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());

                frameBuffer.unbind();

                // END: draw OptiX's output and ImGui.
                // ----------------------------------------------------------------
            }

            // ----------------------------------------------------------------
            // JP: スケーリング

            int32_t display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);

            scaleShader.useProgram();

            glUniform1f(0, UIScaling);

            glActiveTexture(GL_TEXTURE0);
            GLTK::Texture2D &srcFBTex = frameBuffer.getRenderTargetTexture();
            srcFBTex.bind();
            scaleSampler.bindToTextureUnit(0);

            vertexArrayForFullScreen.bind();
            glDrawArrays(GL_TRIANGLES, 0, 3);
            vertexArrayForFullScreen.unbind();

            srcFBTex.unbind();

            // END: scaling
            // ----------------------------------------------------------------

            glfwSwapBuffers(window);

            ++g_frameIndex;
        }

        scaleSampler.finalize();
        scaleShader.finalize();
        frameBuffer.finalize();

        drawOptiXResultShader.finalize();
        outputTexture.finalize();
        outputBufferGL.finalize();

        vertexArrayForFullScreen.finalize();

        ImGui_ImplGlfwGL3_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    else {
        uint32_t renderTargetSizeX = renderImageSizeX;
        uint32_t renderTargetSizeY = renderImageSizeY;

        context->bindOutputBuffer(renderTargetSizeX, renderTargetSizeY, 0);

        VLRDebugPrintf("Setup: %g[s]\n", swGlobal.elapsed(StopWatch::Milliseconds) * 1e-3f);
        swGlobal.start();

        uint32_t numAccumFrames = 0;
        uint32_t imgIndex = 0;
        uint32_t deltaTime = 15 * 1000;
        uint32_t nextTimeToOutput = deltaTime;
        uint32_t finishTime = 123 * 1000 - 3000;
        auto data = new uint32_t[renderTargetSizeX * renderTargetSizeY];
        while (true) {
            context->render(shot.scene, shot.camera, 1, numAccumFrames == 0 ? true : false, &numAccumFrames);

            uint64_t elapsed = swGlobal.elapsed(StopWatch::Milliseconds);
            bool finish = swGlobal.elapsedFromRoot(StopWatch::Milliseconds) > finishTime;
            if (elapsed > nextTimeToOutput || finish) {
                auto output = (const RGBSpectrum*)context->mapOutputBuffer();

                for (int y = 0; y < renderTargetSizeY; ++y) {
                    for (int x = 0; x < renderTargetSizeX; ++x) {
                        RGBSpectrum srcPix = output[y * renderTargetSizeX + x];
                        uint32_t &pix = data[y * renderTargetSizeX + x];

                        srcPix *= g_brightnessCoeff;
                        srcPix = RGBSpectrum::One() - exp(-srcPix);
                        srcPix = sRGB_gamma(srcPix);

                        pix = ((std::min<uint8_t>(srcPix.r * 256, 255) << 0) |
                               (std::min<uint8_t>(srcPix.g * 256, 255) << 8) |
                               (std::min<uint8_t>(srcPix.b * 256, 255) << 16) |
                               (0xFF << 24));
                    }
                }

                char filename[256];
                //sprintf(filename, "%03u.png", imgIndex++);
                //stbi_write_png(filename, renderTargetSizeX, renderTargetSizeY, 4, data, sizeof(data[0]) * renderTargetSizeX);
                sprintf(filename, "%03u.bmp", imgIndex++);
                stbi_write_bmp(filename, renderTargetSizeX, renderTargetSizeY, 4, data);
                VLRDebugPrintf("%u [spp]: %s, %g [s]\n", numAccumFrames, filename, elapsed * 1e-3f);

                context->unmapOutputBuffer();

                if (finish)
                    break;

                nextTimeToOutput += deltaTime;
                nextTimeToOutput = std::min(nextTimeToOutput, finishTime);
            }
        }
        delete[] data;

        swGlobal.stop();

        VLRDebugPrintf("Finish!!: %g[s]\n", swGlobal.stop(StopWatch::Milliseconds) * 1e-3f);
    }

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (optix::Exception ex) {
        VLRDebugPrintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
    }

    return 0;
}