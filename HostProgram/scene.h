#pragma once

#include "common.h"

#include <VLR/vlrcpp.h>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

struct SurfaceMaterialAttributeTuple {
    vlr::SurfaceMaterialRef material;
    vlr::ShaderNodePlug nodeNormal;
    vlr::ShaderNodePlug nodeTangent;
    vlr::ShaderNodePlug nodeAlpha;

    SurfaceMaterialAttributeTuple(const vlr::SurfaceMaterialRef &_material,
                                  const vlr::ShaderNodePlug &_nodeNormal,
                                  const vlr::ShaderNodePlug &_nodeTangent,
                                  const vlr::ShaderNodePlug &_nodeAlpha) :
        material(_material), nodeNormal(_nodeNormal), nodeTangent(_nodeTangent), nodeAlpha(_nodeAlpha) {}
};

struct MeshAttributeTuple {
    bool visible;

    MeshAttributeTuple(bool _visible) : visible(_visible) {}
};

typedef SurfaceMaterialAttributeTuple(*CreateMaterialFunction)(const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &);
typedef MeshAttributeTuple(*PerMeshFunction)(const aiMesh* mesh);

SurfaceMaterialAttributeTuple createMaterialDefaultFunction(const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix);

MeshAttributeTuple perMeshDefaultFunction(const aiMesh* mesh);

static void construct(const vlr::ContextRef &context, const std::string &filePath, bool flipWinding, bool flipV, vlr::InternalNodeRef* nodeOut,
                      CreateMaterialFunction matFunc = createMaterialDefaultFunction, PerMeshFunction meshFunc = perMeshDefaultFunction);



struct Shot {
    vlr::SceneRef scene;

    uint32_t renderTargetSizeX;
    uint32_t renderTargetSizeY;

    float brightnessCoeff;
    float environmentRotation;

    std::vector<vlr::CameraRef> viewpoints;
};

void createScene(const vlr::ContextRef &context, Shot* shot);
