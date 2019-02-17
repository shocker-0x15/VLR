#pragma once

#include "common.h"

#include <VLR/VLRCpp.h>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

VLRCpp::Image2DRef loadImage2D(const VLRCpp::ContextRef &context, const std::string &filepath, bool applyDegamma);

struct SurfaceMaterialAttributeTuple {
	VLRCpp::SurfaceMaterialRef material;
	VLRCpp::ShaderNodeSocket nodeNormal;
	VLRCpp::ShaderNodeSocket nodeAlpha;

	SurfaceMaterialAttributeTuple(const VLRCpp::SurfaceMaterialRef &_material, const VLRCpp::ShaderNodeSocket &_nodeNormal, const VLRCpp::ShaderNodeSocket &_nodeAlpha) :
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

void recursiveConstruct(const VLRCpp::ContextRef &context, const aiScene* objSrc, const aiNode* nodeSrc,
						const std::vector<SurfaceMaterialAttributeTuple> &matAttrTuples, bool flipV, const PerMeshFunction &meshFunc,
						VLRCpp::InternalNodeRef* nodeOut);

static void construct(const VLRCpp::ContextRef &context, const std::string &filePath, bool flipV, VLRCpp::InternalNodeRef* nodeOut,
					  CreateMaterialFunction matFunc = createMaterialDefaultFunction, PerMeshFunction meshFunc = perMeshDefaultFunction);



struct Shot {
	VLRCpp::SceneRef scene;

	uint32_t renderTargetSizeX;
	uint32_t renderTargetSizeY;

	VLR::Point3D cameraPos;
	VLR::Quaternion cameraOrientation;
	float brightnessCoeff;

	VLRCpp::PerspectiveCameraRef perspectiveCamera;
	float persSensitivity;
	float fovYInDeg;
	float lensRadius;
	float objPlaneDistance;

	VLRCpp::EquirectangularCameraRef equirectangularCamera;
	float equiSensitivity;
	float phiAngle;
	float thetaAngle;

	int32_t cameraType;
	VLRCpp::CameraRef camera;
};

void createScene(const VLRCpp::ContextRef &context, Shot* shot);
