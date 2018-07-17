#include <cstdio>
#include <cstdint>

#define NOMINMAX

#include "vdb.h"

#include <VLR/VLR.h>

// DELETE ME
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"
#include <random>

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

//static void recursiveConstruct(const aiScene* objSrc, const aiNode* nodeSrc,
//                               const std::vector<SurfaceMaterialRef> &materials, const std::vector<NormalTextureRef> &normalMaps, const std::vector<FloatTextureRef> &alphaMaps,
//                               VLR::NodeRef* nodeOut) {
//    using namespace VLR;
//
//    if (nodeSrc->mNumMeshes == 0 && nodeSrc->mNumChildren == 0) {
//        nodeOut = nullptr;
//        return;
//    }
//
//    const aiMatrix4x4 &tf = nodeSrc->mTransformation;
//    float tfElems[] = {
//        tf.a1, tf.a2, tf.a3, tf.a4,
//        tf.b1, tf.b2, tf.b3, tf.b4,
//        tf.c1, tf.c2, tf.c3, tf.c4,
//        tf.d1, tf.d2, tf.d3, tf.d4,
//    };
//
//    *nodeOut = createShared<InternalNode>(createShared<StaticTransform>(Matrix4x4(tfElems)));
//
//    std::vector<Triangle> meshIndices;
//    for (int m = 0; m < nodeSrc->mNumMeshes; ++m) {
//        const aiMesh* mesh = objSrc->mMeshes[nodeSrc->mMeshes[m]];
//        if (mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
//            debugPrintf("ignored non triangle mesh.\n");
//            continue;
//        }
//
//        auto surfMesh = createShared<TriangleMeshSurfaceNode>();
//        const SurfaceMaterialRef &surfMat = materials[mesh->mMaterialIndex];
//        const NormalTextureRef &normalMap = normalMaps[mesh->mMaterialIndex];
//        const FloatTextureRef &alphaMap = alphaMaps[mesh->mMaterialIndex];
//
//        std::vector<Vertex> vertices;
//        for (int v = 0; v < mesh->mNumVertices; ++v) {
//            const aiVector3D &p = mesh->mVertices[v];
//            const aiVector3D &n = mesh->mNormals[v];
//            float tangent[3];
//            if (mesh->mTangents == nullptr)
//                makeTangent(n.x, n.y, n.z, tangent);
//            const aiVector3D &t = mesh->mTangents ? mesh->mTangents[v] : aiVector3D(tangent[0], tangent[1], tangent[2]);
//            const aiVector3D &uv = mesh->mNumUVComponents[0] > 0 ? mesh->mTextureCoords[0][v] : aiVector3D(0, 0, 0);
//
//            Vertex outVtx{ Point3D(p.x, p.y, p.z), Normal3D(n.x, n.y, n.z), Vector3D(t.x, t.y, t.z), TexCoord2D(uv.x, uv.y) };
//            float dotNT = dot(outVtx.normal, outVtx.tangent);
//            if (std::fabs(dotNT) >= 0.01f)
//                outVtx.tangent = normalize(outVtx.tangent - dotNT * outVtx.normal);
//            //SLRAssert(absDot(outVtx.normal, outVtx.tangent) < 0.01f, "shading normal and tangent must be orthogonal: %g", absDot(outVtx.normal, outVtx.tangent));
//            vertices.push_back(outVtx);
//        }
//        surfMesh->addVertices(vertices.data(), vertices.size());
//
//        meshIndices.clear();
//        for (int f = 0; f < mesh->mNumFaces; ++f) {
//            const aiFace &face = mesh->mFaces[f];
//            meshIndices.emplace_back(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
//        }
//        surfMesh->addMaterialGroup(surfMat, normalMap, alphaMap, std::move(meshIndices));
//
//        surfMesh->setName(mesh->mName.C_Str());
//
//        surfMesh->useOnlyForBoundary(!meshAttr.render);
//        surfMesh->setAxisForRadialTangent(meshAttr.axisForRadialTangent);
//
//        nodeOut->addChildNode(surfMesh);
//    }
//
//    if (nodeSrc->mNumChildren) {
//        for (int c = 0; c < nodeSrc->mNumChildren; ++c) {
//            InternalNodeRef subNode;
//            recursiveConstruct(objSrc, nodeSrc->mChildren[c], materials, normalMaps, alphaMaps, meshCallback, subNode);
//            if (subNode != nullptr)
//                nodeOut->addChildNode(subNode);
//        }
//    }
//}
//
//static SurfaceAttributeTuple createMaterialDefaultFunction(const aiMaterial* aiMat, const std::string &pathPrefix) {
//    using namespace SLR;
//    aiReturn ret;
//    (void)ret;
//    aiString strValue;
//    float color[3];
//
//    aiMat->Get(AI_MATKEY_NAME, strValue);
//
//    const Texture2DMappingRef &mapping = Texture2DMapping::sharedInstanceRef();
//
//    SpectrumTextureRef diffuseTex;
//    if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
//        Image2DRef image = createImage2D((pathPrefix + strValue.C_Str()).c_str(), ImageStoreMode::AsIs, SpectrumType::Reflectance, false);
//        diffuseTex = createShared<ImageSpectrumTexture>(mapping, image);
//    }
//    else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
//        AssetSpectrumRef sp = Spectrum::create(SpectrumType::Reflectance, ColorSpace::sRGB_NonLinear, color[0], color[1], color[2]);
//        diffuseTex = createShared<ConstantSpectrumTexture>(sp);
//    }
//    else {
//        AssetSpectrumRef sp = Spectrum::create(SpectrumType::Reflectance, ColorSpace::sRGB_NonLinear, 1.0f, 0.0f, 1.0f);
//        diffuseTex = createShared<ConstantSpectrumTexture>(sp);
//    }
//
//    SurfaceMaterialRef mat = SurfaceMaterial::createMatte(diffuseTex, nullptr);
//
//    NormalTextureRef normalTex;
//    if (aiMat->Get(AI_MATKEY_TEXTURE_DISPLACEMENT(0), strValue) == aiReturn_SUCCESS) {
//        Image2DRef image = createImage2D((pathPrefix + strValue.C_Str()).c_str(), ImageStoreMode::NormalTexture, SpectrumType::Reflectance, false);
//        normalTex = createShared<ImageNormalTexture>(mapping, image);
//    }
//
//    FloatTextureRef alphaTex;
//    if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
//        Image2DRef image = createImage2D((pathPrefix + strValue.C_Str()).c_str(), ImageStoreMode::AlphaTexture, SpectrumType::Reflectance, false);
//        alphaTex = createShared<ImageFloatTexture>(mapping, image);
//    }
//
//    return SurfaceAttributeTuple(mat, normalTex, alphaTex);
//}
//
//static void construct(const std::string &filePath, VLR::NodeRef* nodeOut) {
//    using namespace VLR;
//
//    Assimp::Importer importer;
//    const aiScene* scene = importer.ReadFile(filePath, 0);
//    if (!scene) {
//        debugPrintf("Failed to load %s.\n", filePath.c_str());
//        return;
//    }
//    debugPrintf("Reading: %s done.\n", filePath.c_str());
//
//    std::string pathPrefix = filePath.substr(0, filePath.find_last_of("/") + 1);
//
//    // create materials
//    std::vector<SurfaceMaterialRef> materials;
//    std::vector<NormalTextureRef> normalMaps;
//    std::vector<FloatTextureRef> alphaMaps;
//    for (int m = 0; m < scene->mNumMaterials; ++m) {
//        const aiMaterial* aiMat = scene->mMaterials[m];
//
//        SurfaceAttributeTuple surfAttr = materialFunc(aiMat, pathPrefix);
//        materials.push_back(surfAttr.material);
//        normalMaps.push_back(surfAttr.normalMap);
//        alphaMaps.push_back(surfAttr.alphaMap);
//    }
//
//    recursiveConstruct(scene, scene->mRootNode, materials, normalMaps, alphaMaps, nodeOut);
//
//    debugPrintf("Constructing: %s done.\n", filePath.c_str());
//}



int32_t main(int32_t argc, const char* argv[]) {
    using namespace VLRCpp;
    using namespace VLR;

    //try {

    //}
    //catch (optix::Exception ex) {
    //    VLRDebugPrintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
    //}

    VLRCpp::Context context;

    SceneRef scene = context.createScene(std::make_shared<StaticTransform>(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef nodeA = context.createInternalNode("A", createShared<StaticTransform>(translate(-5.0f, 0.0f, 0.0f)));
    InternalNodeRef nodeB = context.createInternalNode("B", createShared<StaticTransform>(translate(5.0f, 0.0f, 0.0f)));

    scene->addChild(nodeA);
    scene->addChild(nodeB);

    InternalNodeRef nodeC = context.createInternalNode("C", createShared<StaticTransform>(translate(0.0f, 0.0f, 0.0f)));

    nodeA->addChild(nodeC);
    nodeB->addChild(nodeC);

    InternalNodeRef nodeD = context.createInternalNode("D", createShared<StaticTransform>(translate(0.0f, 0.0f, -5.0f)));
    InternalNodeRef nodeE = context.createInternalNode("E", createShared<StaticTransform>(translate(0.0f, 0.0f, 5.0f)));

    nodeC->addChild(nodeD);
    nodeC->addChild(nodeE);

    TriangleMeshSurfaceNodeRef meshA = context.createTriangleMeshSurfaceNode("MeshA");
    {
        std::vector<Vertex> vertices;
        vertices.push_back(Vertex{ Point3D(-1.0f, -1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.0f, -1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.0f,  1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.0f,  1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
        meshA->setVertices(vertices.data(), vertices.size());

        Image2DRef image;
        {
            int32_t width, height, n;
            uint8_t* imageData = stbi_load("resources/R.bmp", &width, &height, &n, 4);
            image = context.createLinearImage2D(width, height, DataFormat::RGBA8x4, imageData);
            stbi_image_free(imageData);
        }
        Float4TextureRef texAlbedoRoughness = context.createImageFloat4Texture(image);
        SurfaceMaterialRef matMatte = context.createMatteSurfaceMaterial(texAlbedoRoughness);

        std::vector<uint32_t> matGroup = { 0, 1, 2, 0, 2, 3 };
        meshA->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte);
    }

    nodeD->addChild(meshA);

    scene->test();

    return 0;
}