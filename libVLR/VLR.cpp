#pragma once

#include <VLR.h>
#include "scene_private.h"

// DELETE ME
#define STB_IMAGE_IMPLEMENTATION
#include "../HostProgram/ext/include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "../HostProgram/ext/include/stb_image_write.h"
#include <random>



VLR_API VLRResult vlrCreateContext(VLRContext* context) {
    *context = new VLR::Context;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrDestroyContext(VLRContext context) {
    delete context;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrLinearImage2DCreate(VLRContext context, VLRLinearImage2D* image,
                                         uint32_t width, uint32_t height, VLRDataFormat format, uint8_t* linearData) {
    *image = new VLR::LinearImage2D(*context, linearData, width, height, format);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DDestroy(VLRContext context, VLRLinearImage2D image) {
    if (!image->getType().is(VLR::ObjectType::E_LinearImage2D))
        return VLR_ERROR_INVALID_TYPE;
    delete image;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DGetWidth(VLRLinearImage2D image, uint32_t* width) {
    if (!image->getType().is(VLR::ObjectType::E_LinearImage2D))
        return VLR_ERROR_INVALID_TYPE;
    *width = image->getWidth();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DGetHeight(VLRLinearImage2D image, uint32_t* height) {
    if (!image->getType().is(VLR::ObjectType::E_LinearImage2D))
        return VLR_ERROR_INVALID_TYPE;
    *height = image->getHeight();

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrLinearImage2DGetStride(VLRLinearImage2D image, uint32_t* stride) {
    if (!image->getType().is(VLR::ObjectType::E_LinearImage2D))
        return VLR_ERROR_INVALID_TYPE;
    *stride = image->getStride();

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrImageFloat4TextureCreate(VLRContext context, VLRImageFloat4Texture* texture,
                                              VLRImage2D image) {
    *texture = new VLR::ImageFloat4Texture(*context, image);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrImageFloat4TextureDestroy(VLRContext context, VLRImageFloat4Texture texture) {
    if (!texture->getType().is(VLR::ObjectType::E_ImageFloat4Texture))
        return VLR_ERROR_INVALID_TYPE;
    delete texture;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrMatteSurfaceMaterialCreate(VLRContext context, VLRMatteSurfaceMaterial* material,
                                                VLRFloat4Texture texAlbedoRoughness) {
    *material = new VLR::MatteSurfaceMaterial(*context, texAlbedoRoughness);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrMatteSurfaceMaterialDestroy(VLRContext context, VLRMatteSurfaceMaterial material) {
    if (!material->getType().is(VLR::ObjectType::E_MatteSurfaceMaterial))
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrUE4SurfaceMaterialCreate(VLRContext context, VLRUE4SurfaceMaterial* material,
                                              VLRFloat3Texture texBaseColor, VLRFloat2Texture texRoughnessMetallic) {
    *material = new VLR::UE4SurfaceMaterial(*context, texBaseColor, texRoughnessMetallic);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrUE4SurfaceMaterialDestroy(VLRContext context, VLRUE4SurfaceMaterial material) {
    if (!material->getType().is(VLR::ObjectType::E_UE4SurfaceMaterial))
        return VLR_ERROR_INVALID_TYPE;
    delete material;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrTriangleMeshSurfaceNodeCreate(VLRContext context, VLRTriangleMeshSurfaceNode* surfaceNode, 
                                                   const char* name) {
    *surfaceNode = new VLR::TriangleMeshSurfaceNode(*context, name);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeDestroy(VLRContext context, VLRTriangleMeshSurfaceNode surfaceNode) {
    if (!surfaceNode->getType().is(VLR::ObjectType::E_TriangleMeshSurfaceNode))
        return VLR_ERROR_INVALID_TYPE;
    delete surfaceNode;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeSetVertices(VLRTriangleMeshSurfaceNode surfaceNode, VLRVertex* vertices, uint32_t numVertices) {
    if (!surfaceNode->getType().is(VLR::ObjectType::E_TriangleMeshSurfaceNode))
        return VLR_ERROR_INVALID_TYPE;

    std::vector<VLR::Vertex> vecVertices;
    vecVertices.resize(numVertices);
    std::copy_n(vertices, numVertices, vecVertices.data());

    surfaceNode->setVertices(std::move(vecVertices));

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrTriangleMeshSurfaceNodeAddMaterialGroup(VLRTriangleMeshSurfaceNode surfaceNode, uint32_t* indices, uint32_t numIndices, VLRSurfaceMaterial material) {
    if (!surfaceNode->getType().is(VLR::ObjectType::E_TriangleMeshSurfaceNode))
        return VLR_ERROR_INVALID_TYPE;

    std::vector<uint32_t> vecIndices;
    vecIndices.resize(numIndices);
    std::copy_n(indices, numIndices, vecIndices.data());

    VLR::ObjectType objType = ((VLR::Object*)material)->getType();
    if (!objType.isMemberOf(VLR::ObjectType::E_SurfaceMaterial))
        return VLR_ERROR_INVALID_TYPE;

    surfaceNode->addMaterialGroup(std::move(vecIndices), material);

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrInternalNodeCreate(VLRContext context, VLRInternalNode* node,
                                        const char* name, const VLRStaticTransform* transform) {
    *node = new VLR::InternalNode(*context, name, transform);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeDestroy(VLRContext context, VLRInternalNode node) {
    if (!node->getType().is(VLR::ObjectType::E_InternalNode))
        return VLR_ERROR_INVALID_TYPE;
    delete node;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeSetTransform(VLRInternalNode node, const VLRStaticTransform* localToWorld) {
    if (!node->getType().is(VLR::ObjectType::E_InternalNode))
        return VLR_ERROR_INVALID_TYPE;
    node->setTransform(localToWorld);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeAddChild(VLRInternalNode node, VLRObject child) {
    if (!node->getType().is(VLR::ObjectType::E_InternalNode))
        return VLR_ERROR_INVALID_TYPE;

    VLR::ObjectType objType = ((VLR::Object*)child)->getType();
    if (objType.isMemberOf(VLR::ObjectType::E_InternalNode))
        node->addChild((VLR::InternalNode*)child);
    else if (objType.isMemberOf(VLR::ObjectType::E_SurfaceNode))
        node->addChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrInternalNodeRemoveChild(VLRInternalNode node, VLRObject child) {
    if (!node->getType().is(VLR::ObjectType::E_InternalNode))
        return VLR_ERROR_INVALID_TYPE;

    VLR::ObjectType objType = ((VLR::Object*)child)->getType();
    if (objType.isMemberOf(VLR::ObjectType::E_InternalNode))
        node->removeChild((VLR::InternalNode*)child);
    else if (objType.isMemberOf(VLR::ObjectType::E_SurfaceNode))
        node->removeChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}



VLR_API VLRResult vlrSceneCreate(VLRContext context, VLRScene* scene,
                                 const VLRStaticTransform* transform) {
    *scene = new VLR::Scene(*context, transform);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneDestroy(VLRContext context, VLRScene scene) {
    if (!scene->getType().is(VLR::ObjectType::E_Scene))
        return VLR_ERROR_INVALID_TYPE;
    delete scene;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneSetTransform(VLRScene scene, const VLRStaticTransform* localToWorld) {
    if (!scene->getType().is(VLR::ObjectType::E_Scene))
        return VLR_ERROR_INVALID_TYPE;
    scene->setTransform(localToWorld);

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneAddChild(VLRScene scene, VLRObject child) {
    if (!scene->getType().is(VLR::ObjectType::E_Scene))
        return VLR_ERROR_INVALID_TYPE;

    VLR::ObjectType objType = ((VLR::Object*)child)->getType();
    if (objType.isMemberOf(VLR::ObjectType::E_InternalNode))
        scene->addChild((VLR::InternalNode*)child);
    else if (objType.isMemberOf(VLR::ObjectType::E_SurfaceNode))
        scene->addChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneRemoveChild(VLRScene scene, VLRObject child) {
    if (!scene->getType().is(VLR::ObjectType::E_Scene))
        return VLR_ERROR_INVALID_TYPE;

    VLR::ObjectType objType = ((VLR::Object*)child)->getType();
    if (objType.isMemberOf(VLR::ObjectType::E_InternalNode))
        scene->removeChild((VLR::InternalNode*)child);
    else if (objType.isMemberOf(VLR::ObjectType::E_SurfaceNode))
        scene->removeChild((VLR::SurfaceNode*)child);
    else
        return VLR_ERROR_INVALID_TYPE;

    return VLR_ERROR_NO_ERROR;
}

VLR_API VLRResult vlrSceneTest(VLRScene scene) {
    try {
        using namespace VLR;

        Context& context = scene->getContext();
        optix::Context &optixContext = context.getOptiXContext();

        SHGroup &shGroup = scene->getSHGroup();

        optixContext["VLR::pv_topGroup"]->set(shGroup.getOptiXObject());

        optix::Buffer rngBuffer = optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1280, 720);
        rngBuffer->setElementSize(sizeof(uint64_t));
        {
            std::mt19937_64 rng(591842031321323413);

            auto dstData = (uint64_t*)rngBuffer->map();
            for (int y = 0; y < 720; ++y) {
                for (int x = 0; x < 1280; ++x) {
                    dstData[y * 1280 + x] = rng();
                }
            }
            rngBuffer->unmap();
        }
        optixContext["VLR::pv_rngBuffer"]->set(rngBuffer);

        optix::Buffer outputBuffer = optixContext->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1280, 720);
        outputBuffer->setElementSize(sizeof(RGBSpectrum));
        {
            auto dstData = (RGBSpectrum*)outputBuffer->map();
            std::fill_n(dstData, 1280 * 720, RGBSpectrum::Zero());
            outputBuffer->unmap();
        }
        optixContext["VLR::pv_outputBuffer"]->set(outputBuffer);

        optixContext->setPrintEnabled(true);
        optixContext->setPrintBufferSize(4096);

        Shared::ThinLensCamera thinLensParams(1280.0f / 720.0f, 40 * M_PI / 180, 0.0f, 1.0f, 1.0f);
        thinLensParams.position = Point3D(0, 0, 15);
        thinLensParams.orientation = qRotateY<float>(M_PI);// *qRotateX<float>(45 * M_PI / 180);
        optixContext["VLR::pv_thinLensCamera"]->setUserData(sizeof(Shared::ThinLensCamera), &thinLensParams);

        shGroup.printOptiXHierarchy();

        optixContext->validate();



        //rootNode->setTransform(createShared<StaticTransform>(translate(0.0f, 5.0f, 0.0f)));
        //nodeC->setTransform(createShared<StaticTransform>(translate(0.0f, -10.0f, 0.0f)));

        //rootNode->removeChild(nodeB);

        optixContext->launch(0, 1280, 720);

        {
            auto srcData = (const RGBSpectrum*)outputBuffer->map();
            auto dstData = new uint8_t[1280 * 720 * 3];

            for (int y = 0; y < 720; ++y) {
                for (int x = 0; x < 1280; ++x) {
                    const RGBSpectrum &src = srcData[y * 1280 + x];
                    uint8_t* dst = dstData + (y * 1280 + x) * 3;

                    const auto quantize = [](float x) {
                        uint32_t qv = (uint32_t)(256 * x);
                        return (uint8_t)std::min<uint32_t>(255, qv);
                    };

                    dst[0] = quantize(src.r);
                    dst[1] = quantize(src.g);
                    dst[2] = quantize(src.b);
                }
            }

            stbi_write_png("output.png", 1280, 720, 3, dstData, 1280 * 3);

            delete[] dstData;
            outputBuffer->unmap();
        }
    }
    catch (optix::Exception ex) {
        VLRDebugPrintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
    }

    return VLR_ERROR_NO_ERROR;
}
