#pragma once

#include "common.h"

#include <VLR/VLRCpp.h>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

struct SurfaceMaterialAttributeTuple {
    VLRCpp::SurfaceMaterialRef material;
    VLRCpp::ShaderNodePlug nodeNormal;
    VLRCpp::ShaderNodePlug nodeAlpha;

    SurfaceMaterialAttributeTuple(const VLRCpp::SurfaceMaterialRef &_material, const VLRCpp::ShaderNodePlug &_nodeNormal, const VLRCpp::ShaderNodePlug &_nodeAlpha) :
        material(_material), nodeNormal(_nodeNormal), nodeAlpha(_nodeAlpha) {}
};

struct MeshAttributeTuple {
    bool visible;
    VLRTangentType tangentType;

    MeshAttributeTuple(bool _visible, VLRTangentType _tangentType) : visible(_visible), tangentType(_tangentType) {}
};

typedef SurfaceMaterialAttributeTuple(*CreateMaterialFunction)(const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &);
typedef MeshAttributeTuple(*PerMeshFunction)(const aiMesh* mesh);

SurfaceMaterialAttributeTuple createMaterialDefaultFunction(const VLRCpp::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix);

MeshAttributeTuple perMeshDefaultFunction(const aiMesh* mesh);

static void construct(const VLRCpp::ContextRef &context, const std::string &filePath, bool flipWinding, bool flipV, VLRCpp::InternalNodeRef* nodeOut,
                      CreateMaterialFunction matFunc = createMaterialDefaultFunction, PerMeshFunction meshFunc = perMeshDefaultFunction);



struct Shot {
    VLRCpp::SceneRef scene;

    uint32_t renderTargetSizeX;
    uint32_t renderTargetSizeY;

    float brightnessCoeff;
    float environmentRotation;

    std::vector<VLRCpp::CameraRef> viewpoints;
};

void createScene(const VLRCpp::ContextRef &context, Shot* shot);
