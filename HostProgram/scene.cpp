#include "scene.h"

SurfaceMaterialAttributeTuple createMaterialDefaultFunction(const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
    using namespace vlr;

    aiReturn ret;
    (void)ret;
    aiString strValue;
    float color[3];

    aiMat->Get(AI_MATKEY_NAME, strValue);
    hpprintf("Material: %s\n", strValue.C_Str());

    SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
    ShaderNodePlug plugNormal;
    ShaderNodePlug plugTangent;
    ShaderNodePlug plugAlpha;

    Image2DRef imgDiffuse;
    ShaderNodeRef texDiffuse;
    Image2DRef imgNormal;
    ShaderNodeRef texNormal;
    Image2DRef imgAlpha;
    ShaderNodeRef texAlpha;
    
    // Base Color
    if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
        texDiffuse = context->createShaderNode("Image2DTexture");
        imgDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
        texDiffuse->set("image", imgDiffuse);
        mat->set("albedo", texDiffuse->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    }
    else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
        mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", color[0], color[1], color[2] });
    }
    else {
        mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 1.0f, 0.0f, 1.0f });
    }

    // Normal
    if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS) {
        imgNormal = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
        texNormal = context->createShaderNode("Image2DTexture");
        texNormal->set("image", imgNormal);
    }

    // Alpha
    if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
        imgAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
        texAlpha = context->createShaderNode("Image2DTexture");
        texAlpha->set("image", imgAlpha);
    }

    if (imgNormal)
        plugNormal = texNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);

    if (imgAlpha) {
        if (std::strcmp(imgAlpha->getOriginalDataFormat(), "Gray8") == 0)
            plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_float1, 0);
        else
            plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_Alpha, 0);
    }
    else if (imgDiffuse && imgDiffuse->originalHasAlpha()) {
        plugAlpha = texDiffuse->getPlug(VLRShaderNodePlugType_Alpha, 0);
    }

    return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
}

MeshAttributeTuple perMeshDefaultFunction(const aiMesh* mesh) {
    return MeshAttributeTuple(true);
}

void recursiveConstruct(const vlr::ContextRef &context, const aiScene* objSrc, const aiNode* nodeSrc,
                        const std::vector<SurfaceMaterialAttributeTuple> &matAttrTuples, const PerMeshFunction &meshFunc,
                        vlr::InternalNodeRef* nodeOut) {
    using namespace vlr;

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
        const ShaderNodePlug &nodeNormal = attrTuple.nodeNormal;
        const ShaderNodePlug &nodeTangent = attrTuple.nodeTangent;
        const ShaderNodePlug &nodeAlpha = attrTuple.nodeAlpha;

        std::vector<Vertex> vertices;
        for (int v = 0; v < mesh->mNumVertices; ++v) {
            const aiVector3D &p = mesh->mVertices[v];
            const aiVector3D &n = mesh->mNormals[v];
            Vector3D tangent, bitangent;
            if (mesh->mTangents == nullptr)
                Normal3D(n.x, n.y, n.z).makeCoordinateSystem(&tangent, &bitangent);
            aiVector3D t(NAN, NAN, NAN);
            if (mesh->mTangents)
                t = mesh->mTangents[v];
            if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.z))
                t = aiVector3D(tangent[0], tangent[1], tangent[2]);
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
        surfMesh->addMaterialGroup(meshIndices.data(), meshIndices.size(), surfMat, nodeNormal, nodeTangent, nodeAlpha);

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

void construct(const vlr::ContextRef &context, const std::string &filePath, bool flipWinding, bool flipV, vlr::InternalNodeRef* nodeOut,
               CreateMaterialFunction matFunc, PerMeshFunction meshFunc) {
    using namespace vlr;

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



void printParameterInfos(const vlr::QueryableRef& connectable) {
    uint32_t numParams = connectable->getNumParameters();
    for (int i = 0; i < numParams; ++i) {
        auto paramInfo = connectable->getParameterInfo(i);

        hpprintf("%u: name: %s\n", i, paramInfo.getName());
        hpprintf("   type: %s, size: %u, flags: %u\n", paramInfo.getType(), paramInfo.getTupleSize(), (uint32_t)paramInfo.getFormFlags());
    }
}

#define ASSETS_DIR "resources/assets/"

void createCornellBoxScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

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

        //// Texture Coordinate Direction Check
        //vertices.push_back(Vertex{ Point3D(-0.5f, 2.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        //vertices.push_back(Vertex{ Point3D(-0.5f, 1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        //vertices.push_back(Vertex{ Point3D(0.5f, 1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        //vertices.push_back(Vertex{ Point3D(0.5f, 2.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1,  0,  0), TexCoord2D(1.0f, 0.0f) });

        cornellBox->setVertices(vertices.data(), vertices.size());

        //// Texture Coordinate Direction Check
        //{
        //    //auto nodeAlbedo = context->createShaderNode("Image2DTexture");
        //    //nodeAlbedo->set("image", loadImage2D(context, "resources/mountain_heightmap.png", "Reflectance", "Rec709(D65) sRGB Gamma"));
        //    //nodeAlbedo->setTextureFilterMode(VLRTextureFilter_Nearest, VLRTextureFilter_Nearest);
        //    auto matMatte = context->createSurfaceMaterial("Matte");
        //    //matMatte->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));
        //    matMatte->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.8f, 0.8f, 0.8f });

        //    auto nodeNormal = context->createShaderNode("Image2DTexture");
        //    nodeNormal->set("image", loadImage2D(context, "resources/mountain_heightmap.jpg", "NA", "Rec709(D65)"));
        //    nodeNormal->set("bump type", "Height Map");
        //    nodeNormal->set("min filter", "Nearest");
        //    nodeNormal->set("mag filter", "Nearest");
        //    nodeNormal->set("wrap u", "Clamp to Edge");
        //    nodeNormal->set("wrap v", "Clamp to Edge");

        //    std::vector<uint32_t> matGroup = {
        //        28, 29, 30, 28, 30, 31
        //    };
        //    cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, nodeNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0), ShaderNodePlug(), ShaderNodePlug());
        //}

        {
            auto image = loadImage2D(context, "resources/checkerboard_line.png", "Reflectance", "Rec709(D65) sRGB Gamma");
            auto nodeAlbedo = context->createShaderNode("Image2DTexture");
            nodeAlbedo->set("image", image);
            nodeAlbedo->set("filter", "Nearest");
            auto matMatte = context->createSurfaceMaterial("Matte");
            matMatte->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }

        {
            auto matMatte = context->createSurfaceMaterial("Matte");
            matMatte->set("albedo", VLRImmediateSpectrum{ "Rec709(D65) sRGB Gamma", 0.75f, 0.75f, 0.75f });

            std::vector<uint32_t> matGroup = {
                4, 5, 6, 4, 6, 7,
                8, 9, 10, 8, 10, 11,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }

        {
            auto matMatte = context->createSurfaceMaterial("Matte");
            matMatte->set("albedo", VLRImmediateSpectrum{ "Rec709(D65) sRGB Gamma", 0.75f, 0.25f, 0.25f });

            //float value[3] = { 0.06f, 0.02f, 0.02f };
            //Float3TextureRef texEmittance = context->createConstantFloat3Texture(value);
            //SurfaceMaterialRef matMatte = context->createSurfaceMaterial("DiffuseEmitter");

            std::vector<uint32_t> matGroup = {
                12, 13, 14, 12, 14, 15,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }

        {
            auto matMatte = context->createSurfaceMaterial("Matte");
            matMatte->set("albedo", VLRImmediateSpectrum{ "Rec709(D65) sRGB Gamma", 0.25f, 0.25f, 0.75f });

            std::vector<uint32_t> matGroup = {
                16, 17, 18, 16, 18, 19,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }

        {
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 30.0f, 30.0f, 30.0f });

            std::vector<uint32_t> matGroup = {
                20, 21, 22, 20, 22, 23,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }

        {
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 100.0f, 100.0f, 100.0f });

            std::vector<uint32_t> matGroup = {
                24, 25, 26, 24, 26, 27,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    shot->scene->addChild(cornellBox);



    InternalNodeRef sphereNode;
    construct(context, "resources/sphere/sphere.obj", false, false, &sphereNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        //auto matA = context->createSurfaceMaterial("SpecularReflection");
        //matA->set("coeff", VLRImmediateSpectrum{ "Rec709(D65)", 0.999f, 0.999f, 0.999f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.27579f, 0.940922f, 0.574879f }); // Aluminum
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 7.30257f, 6.33458f, 5.16694f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.237698f, 0.734847f, 1.37062f }); // Copper
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.44233f, 2.55751f, 2.23429f });
        //matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.12481f, 0.468228f, 1.44476f }); // Gold
        //matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.32107f, 2.23761f, 1.69196f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 2.91705f, 2.92092f, 2.53253f }); // Iron
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.06696f, 2.93804f, 2.7429f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.9566f, 1.82777f, 1.46089f }); // Lead
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.49593f, 3.38158f, 3.17737f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.99144f, 1.5186f, 1.00058f }); // Mercury
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 5.25161f, 4.6095f, 3.7646f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 2.32528f, 2.06722f, 1.81479f }); // Platinum
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 4.19238f, 3.67941f, 3.06551f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.157099f, 0.144013f, 0.134847f }); // Silver
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.82431f, 3.1451f, 2.27711f });
        ////matA->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 2.71866f, 2.50954f, 2.22767f }); // Titanium
        ////matA->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.79521f, 3.40035f, 3.00114f });

        auto matA = context->createSurfaceMaterial("SpecularScattering");
        matA->set("coeff", VLRImmediateSpectrum{ "Rec709(D65)", 0.999f, 0.999f, 0.999f });
        matA->set("eta ext", VLRImmediateSpectrum{ "Rec709(D65)", 1.00036f, 1.00021f, 1.00071f }); // Air
        matA->set("eta int", VLRImmediateSpectrum{ "Rec709(D65)", 2.41174f, 2.42343f, 2.44936f }); // Diamond
        //matA->set("eta int", VLRImmediateSpectrum{ "Rec709", 1.33161f, 1.33331f, 1.33799f }); // Water
        //matA->set("eta int", VLRImmediateSpectrum{ "Rec709", 1.51455f, 1.51816f, 1.52642f }); // Glass BK7

        //auto matB = context->createSurfaceMaterial("DiffuseEmitter");
        //matB->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 1, 1, 1 });

        //auto matB = context->createSurfaceMaterial("Matte");
        //matB->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.05f, 0.3f, 0.05f });

        //auto mat = context->createSurfaceMaterial("Multi");
        //mat->set("0", matA);
        //mat->set("1", matB);

        return SurfaceMaterialAttributeTuple(matA, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
    });
    shot->scene->addChild(sphereNode);
    sphereNode->setTransform(context->createStaticTransform(scale(0.5f) * translate<float>(0.0f, 1.0f, 0.0f)));



    //Image2DRef imgEnv = loadImage2D(context, "resources/environments/WhiteOne.exr");
    //Float3TextureRef texEnv = context->createImageFloat3Texture(imgEnv);
    //SurfaceMaterialRef matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    //scene->setEnvironment(matEnv);

    shot->renderTargetSizeX = 1024;
    shot->renderTargetSizeY = 1024;

    shot->brightnessCoeff = 1.0f;
    shot->environmentRotation = 0.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0, 1.5f, 6.0f));
        camera->set("orientation", qRotateY<float>(M_PI));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createMaterialTestScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createShaderNode("ScaleAndOffsetUVTextureMap2D");
        nodeTexCoord->set("offset", offset, 2);
        nodeTexCoord->set("scale", scale, 2);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", "Reflectance", "Rec709(D65) sRGB Gamma");

        ShaderNodeRef nodeAlbedo = context->createShaderNode("Image2DTexture");
        nodeAlbedo->set("image", image);
        nodeAlbedo->set("filter", "Nearest");
        nodeAlbedo->set("texcoord", nodeTexCoord->getPlug(VLRShaderNodePlugType_TextureCoordinates, 0));

        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        mat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
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
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 50.0f, 50.0f, 50.0f });

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, -3.0f) * rotateX<float>(M_PI / 2)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, "resources/material_test/mitsuba_knob.obj", false, false, &modelNode, 
              [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        if (strcmp(strValue.C_Str(), "Base") == 0) {
            SurfaceMaterialRef matteMat = context->createSurfaceMaterial("Matte");
            matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.18f, 0.18f, 0.18f });

            mat = matteMat;
        }
        else if (strcmp(strValue.C_Str(), "Glossy") == 0) {
            Image2DRef imgBaseColorAlpha = loadImage2D(context, pathPrefix + "TexturesCom_Leaves0165_1_alphamasked_S.png", "Reflectance", "Rec709(D65) sRGB Gamma");
            ShaderNodeRef nodeBaseColorAlpha = context->createShaderNode("Image2DTexture");
            nodeBaseColorAlpha->set("image", imgBaseColorAlpha);

            SurfaceMaterialRef ue4Mat = context->createSurfaceMaterial("UE4");
            ue4Mat->set("base color", nodeBaseColorAlpha->getPlug(VLRShaderNodePlugType_Spectrum, 0));
            ue4Mat->set("base color", VLRImmediateSpectrum{ "Rec709(D65) sRGB Gamma", 0.75f, 0.5f, 0.0025f });
            ue4Mat->set("occlusion", 0.0f);
            ue4Mat->set("roughness", 0.3f);
            ue4Mat->set("metallic", 0.0f);

            mat = ue4Mat;

            plugAlpha = nodeBaseColorAlpha->getPlug(VLRShaderNodePlugType_Alpha, 0);

            //auto matteMat = context->createSurfaceMaterial("Matte");
            //float lambdas[] = { 360.0, 418.75, 477.5, 536.25, 595.0, 653.75, 712.5, 771.25, 830.0 };
            //float values[] = { 0.8f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            ////auto nodeAlbedo = context->createShaderNode("RegularSampledSpectrum");
            ////nodeAlbedo->setSpectrum(360.0f, 830.0f, values, lengthof(values));
            //auto nodeAlbedo = context->createShaderNode("IrregularSampledSpectrum");
            //nodeAlbedo->setSpectrum(lambdas, values, lengthof(values));
            //matteMat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));
            //mat = matteMat;

            //SurfaceMaterialRef mfMat = context->createSurfaceMaterial("MicrofacetReflection");
            //// Aluminum
            //mfMat->set("eta", RGBSpectrum(1.27579f, 0.940922f, 0.574879f));
            //mfMat->set("k", RGBSpectrum(7.30257f, 6.33458f, 5.16694f));
            //mfMat->set("roughness", 0.2f);
            //mfMat->setAnisotropy(0.9f);
            //mfMat->setRotation(0.0f);

            //mat = mfMat;

            //GeometryShaderNodeRef nodeGeom = context->createShaderNode("Geometry");
            //Vector3DToSpectrumShaderNodeRef nodeVec2Sp = context->createShaderNode("Vector3DToSpectrum");
            //nodeVec2Sp->setVector3D(nodeGeom->getPlug(VLRShaderNodePlugType_Vector3D, 0));
            //SurfaceMaterialRef mtMat = context->createSurfaceMaterial("Matte");
            //mtMat->set("albedo", nodeVec2Sp->getPlug(VLRShaderNodePlugType_Spectrum, 0));

            //mat = mtMat;
        }

        auto nodeTangent = context->createShaderNode("Tangent");
        nodeTangent->set("tangent type", "Radial Y");
        plugTangent = nodeTangent->getPlug(VLRShaderNodePlugType_float1, 0);

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    },
              [](const aiMesh* mesh) {
        return MeshAttributeTuple(true);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.04089, 0)));



    auto imgEnv = loadImage2D(context, "resources/material_test/Chelsea_Stairs_3k.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 0.1f, 0.1f, 0.1f });
    shot->environmentRotation = -M_PI / 2;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.0f, 5.0f, 10.0f));
        camera->set("orientation", qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createAnisotropyScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createShaderNode("ScaleAndOffsetUVTextureMap2D");
        nodeTexCoord->set("offset", offset, 2);
        nodeTexCoord->set("scale", scale, 2);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", "Reflectance", "Rec709(D65) sRGB Gamma");

        ShaderNodeRef nodeAlbedo = context->createShaderNode("Image2DTexture");
        nodeAlbedo->set("image", image);
        nodeAlbedo->set("filter", "Nearest");
        nodeAlbedo->set("texcoord", nodeTexCoord->getPlug(VLRShaderNodePlugType_TextureCoordinates, 0));

        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        mat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
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
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 150.0f, 150.0f, 150.0f });

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, 0.0f) * rotateX<float>(M_PI)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, ASSETS_DIR"rounded_box/rounded_box_0.fbx", false, true, &modelNode, 
              [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        //if (strcmp(strValue.C_Str(), "Material.001") == 0) {
            auto mfMat = context->createSurfaceMaterial("MicrofacetReflection");
            // Aluminum
            mfMat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.27579f, 0.940922f, 0.574879f });
            mfMat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 7.30257f, 6.33458f, 5.16694f });
            mfMat->set("roughness", 0.2f);
            mfMat->set("anisotropy", 0.9f);
            mfMat->set("rotation", 0.25f);

            mat = mfMat;

            auto image = loadImage2D(context, pathPrefix + "height_test/Height.png", "NA", "Rec709(D65)");
            auto tex = context->createShaderNode("Image2DTexture");
            tex->set("image", image);
            tex->set("bump type", "Height Map");
            tex->set("bump coeff", 1.0f);
            plugNormal = tex->getPlug(VLRShaderNodePlugType_Normal3D, 0);
        //}
        //else {
        //    mat = context->createSurfaceMaterial("Matte");
        //}

        auto nodeTangent = context->createShaderNode("Tangent");
        nodeTangent->set("tangent type", "Radial Z");
        plugTangent = nodeTangent->getPlug(VLRShaderNodePlugType_Vector3D, 0);

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    },
              [](const aiMesh* mesh) {
        return MeshAttributeTuple(true);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(
        translate<float>(0, 1.0f, 0) * scale(0.01f * 0.75f) * 
        rotateX<float>(10 * M_PI / 180) * rotateY<float>(45 * M_PI / 180) * rotateX<float>(15 * M_PI / 180)
    ));



    //auto imgEnv = loadImage2D(context, "resources/material_test/Chelsea_Stairs_3k.exr", "Light Source", "Rec709(D65)");
    //auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    //nodeEnvTex->set("image", imgEnv);
    //auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    //matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    ////matEnv->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 0.1f, 0.1f, 0.1f });
    //shot->environmentRotation = -M_PI / 2;
    //shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1024;
    shot->renderTargetSizeY = 1024;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.0f, 2.5f, 5.0f));
        camera->set("orientation", qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createWhiteFurnaceTestScene(const vlr::ContextRef& context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/mitsuba_knob.obj", false, false, &modelNode,
              [](const vlr::ContextRef& context, const aiMaterial* aiMat, const std::string& pathPrefix) {
                  using namespace vlr;
                  using namespace vlr;

                  aiReturn ret;
                  (void)ret;
                  aiString strValue;
                  float color[3];

                  aiMat->Get(AI_MATKEY_NAME, strValue);

                  SurfaceMaterialRef mat;
                  ShaderNodePlug plugNormal;
                  ShaderNodePlug plugTangent;
                  ShaderNodePlug plugAlpha;
                  if (strcmp(strValue.C_Str(), "Base") == 0) {
                      SurfaceMaterialRef matteMat = context->createSurfaceMaterial("Matte");
                      matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 1, 1, 1 });

                      mat = matteMat;
                  }
                  else if (strcmp(strValue.C_Str(), "Glossy") == 0) {
                      auto mMat = context->createSurfaceMaterial("MicrofacetReflection");
                      mMat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.157099f, 0.144013f, 0.134847f }); // Silver
                      mMat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.82431f, 3.1451f, 2.27711f });
                      mMat->set("roughness", 1.0f);
                      mat = mMat;

                      //auto ue4Mat = context->createSurfaceMaterial("UE4");
                      //ue4Mat->set("base color", VLRImmediateSpectrum{ "Rec709(D65) sRGB Gamma", 1, 1, 1 });
                      //ue4Mat->set("metallic", 0.0f);
                      //ue4Mat->set("roughness", 0.01f);
                      //mat = ue4Mat;
                  }

                  auto nodeTangent = context->createShaderNode("Tangent");
                  nodeTangent->set("tangent type", "Radial Y");
                  plugTangent = nodeTangent->getPlug(VLRShaderNodePlugType_float1, 0);

                  return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
              },
              [](const aiMesh* mesh) {
                  return MeshAttributeTuple(true);
              });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.04089, 0)));



    auto imgEnv = loadImage2D(context, "resources/environments/WhiteOne.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1024;
    shot->renderTargetSizeY = 1024;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.0f, 2.5f, 5.0f));
        camera->set("orientation", qRotateY<float>(M_PI) * qRotateX<float>(19 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createColorCheckerScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

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
            auto spectrum = context->createShaderNode("RegularSampledSpectrum");
            spectrum->set("spectrum type", "Reflectance");
            spectrum->set("min wavelength", MinLambda);
            spectrum->set("max wavelength", MaxLambda);
            spectrum->set("values", ColorCheckerSpectrumValues[i], lengthof(ColorCheckerLambdas));

            auto matMatte = context->createSurfaceMaterial("Matte");
            matMatte->set("albedo", spectrum->getPlug(VLRShaderNodePlugType_Spectrum, 0));

            uint32_t indexOffset = 4 * i;
            std::vector<uint32_t> matGroup = {
                indexOffset + 0, indexOffset + 1, indexOffset + 2,
                indexOffset + 0, indexOffset + 2, indexOffset + 3
            };
            colorChecker->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    auto colorCheckerNode = context->createInternalNode("ColorChecker", context->createStaticTransform(translate<float>(-3.0f, 2.0f, 0.0f) * rotateX<float>(M_PI / 2)));
    colorCheckerNode->addChild(colorChecker);
    shot->scene->addChild(colorCheckerNode);

    //TriangleMeshSurfaceNodeRef light = context->createTriangleMeshSurfaceNode("light");
    //{
    //    std::vector<Vertex> vertices;

    //    // Light
    //    vertices.push_back(Vertex{ Point3D(-1.5f, 0.0f, -1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
    //    vertices.push_back(Vertex{ Point3D(-1.5f, 0.0f, 1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
    //    vertices.push_back(Vertex{ Point3D(1.5f, 0.0f, 1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
    //    vertices.push_back(Vertex{ Point3D(1.5f, 0.0f, -1.5f), Normal3D(0, 1, 0), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });

    //    light->setVertices(vertices.data(), vertices.size());

    //    {
    //        SurfaceMaterialRef matLight = context->createSurfaceMaterial("DiffuseEmitter");
    //        matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 300.0f, 300.0f, 300.0f });

    //        std::vector<uint32_t> matGroup = {
    //            0, 1, 2, 0, 2, 3
    //        };
    //        light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
    //    }
    //}
    //InternalNodeRef lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, 1.5f) * rotateX<float>(M_PI)));
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
    auto spectrum = context->createShaderNode("RegularSampledSpectrum");
    spectrum->set("spectrum type", "Light Source");
    spectrum->set("min wavelength", MinLambda);
    spectrum->set("max wavelength", MaxLambda);
    spectrum->set("values", Values, NumLambdas);
    float envScale = 0.02f;
    //const float IlluminantESpectrumValues[] = { 1.0f, 1.0f };
    //auto spectrum = context->createShaderNode("RegularSampledSpectrum");
    //spectrum->setSpectrum(0.0f, 1000.0f, IlluminantESpectrumValues, 2);
    //float envScale = 1.0f;
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", spectrum->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", VLRImmediateSpectrum{ "xyY", 1.0f / 3, 1.0f / 3, 1.0f });
    matEnv->set("scale", envScale);
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0, 0, 6.0f));
        camera->set("orientation", qRotateY<float>(M_PI));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createColorInterpolationTestScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createShaderNode("ScaleAndOffsetUVTextureMap2D");
        nodeTexCoord->set("offset", offset, 2);
        nodeTexCoord->set("scale", scale, 2);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", "Reflectance", "Rec709(D65) sRGB Gamma");

        ShaderNodeRef nodeAlbedo = context->createShaderNode("Image2DTexture");
        nodeAlbedo->set("image", image);
        nodeAlbedo->set("filter", "Nearest");
        nodeAlbedo->set("texcoord", nodeTexCoord->getPlug(VLRShaderNodePlugType_TextureCoordinates, 0));

        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        mat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
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
            auto image = loadImage2D(context, "resources/material_test/jumping_colors.png", "Reflectance", "Rec709(D65) sRGB Gamma");
            auto nodeAlbedo = context->createShaderNode("Image2DTexture");
            nodeAlbedo->set("image", image);
            nodeAlbedo->set("wrap u", "Clamp to Edge");
            nodeAlbedo->set("wrap v", "Clamp to Edge");
            auto mat = context->createSurfaceMaterial("Matte");
            mat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

            //auto image = loadImage2D(context, "resources/material_test/jumping_colors.png", "Light Source", "Rec709(D65) sRGB Gamma");
            //auto nodeEmittance = context->createShaderNode("Image2DTexture");
            //nodeEmittance->set("image", image);
            //nodeEmittance->setTextureWrapMode(VLRTextureWrapMode_ClampToEdge, VLRTextureWrapMode_ClampToEdge);
            //auto mat = context->createSurfaceMaterial("DiffuseEmitter");
            //mat->set("emittance", nodeEmittance->getPlug(VLRShaderNodePlugType_Spectrum, 0));
            //mat->set("scale", 10.0f);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            colorTestPlate->addMaterialGroup(matGroup.data(), matGroup.size(), mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
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
    auto spectrum = context->createShaderNode("RegularSampledSpectrum");
    spectrum->set("spectrum type", "Light Source");
    spectrum->set("min wavelength", MinLambda);
    spectrum->set("max wavelength", MaxLambda);
    spectrum->set("values", Values, NumLambdas);
    float envScale = 0.02f;
    //const float IlluminantESpectrumValues[] = { 1.0f, 1.0f };
    //auto spectrum = context->createShaderNode("RegularSampledSpectrum");
    //spectrum->setSpectrum(0.0f, 1000.0f, IlluminantESpectrumValues, 2);
    //float envScale = 1.0f;
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", spectrum->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", VLRImmediateSpectrum{ "xyY", 1.0f / 3, 1.0f / 3, 1.0f });
    matEnv->set("scale", envScale);
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.0f, 5.0f, 10.0f));
        camera->set("orientation", qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createSubstanceManScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    construct(context, "resources/material_test/paper.obj", false, true, &modelNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        float offset[2] = { 0, 0 };
        float scale[2] = { 10, 20 };
        auto nodeTexCoord = context->createShaderNode("ScaleAndOffsetUVTextureMap2D");
        nodeTexCoord->set("offset", offset, 2);
        nodeTexCoord->set("scale", scale, 2);

        Image2DRef image = loadImage2D(context, pathPrefix + "grid_80p_white_18p_gray.png", "Reflectance", "Rec709(D65) sRGB Gamma");

        ShaderNodeRef nodeAlbedo = context->createShaderNode("Image2DTexture");
        nodeAlbedo->set("image", image);
        nodeAlbedo->set("filter", "Nearest");
        nodeAlbedo->set("texcoord", nodeTexCoord->getPlug(VLRShaderNodePlugType_TextureCoordinates, 0));

        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        mat->set("albedo", nodeAlbedo->getPlug(VLRShaderNodePlugType_Spectrum, 0));

        return SurfaceMaterialAttributeTuple(mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
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
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 50.0f, 50.0f, 50.0f });

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, -3.0f) * rotateX<float>(M_PI / 2)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, ASSETS_DIR"spman2/spman2.obj", false, true, &modelNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        if (strcmp(strValue.C_Str(), "_Head1") == 0) {
            auto nodeBaseColor = context->createShaderNode("Image2DTexture");
            nodeBaseColor->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_BaseColor.png", "Reflectance", "Rec709(D65) sRGB Gamma"));
            auto nodeORM = context->createShaderNode("Image2DTexture");
            nodeORM->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_OcclusionRoughnessMetallic.png", "NA", "Rec709(D65)"));
            auto nodeNormal = context->createShaderNode("Image2DTexture");
            nodeNormal->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_01_Head_NormalAlpha.png", "NA", "Rec709(D65)"));

            auto ue4Mat = context->createSurfaceMaterial("UE4");
            ue4Mat->set("base color", nodeBaseColor->getPlug(VLRShaderNodePlugType_Spectrum, 0));
            ue4Mat->set("occlusion/roughness/metallic", nodeORM->getPlug(VLRShaderNodePlugType_float3, 0));

            mat = ue4Mat;
            plugNormal = nodeNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);
        }
        else if (strcmp(strValue.C_Str(), "_Body1") == 0) {
            auto nodeBaseColor = context->createShaderNode("Image2DTexture");
            nodeBaseColor->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_BaseColor.png", "Reflectance", "Rec709(D65) sRGB Gamma"));
            auto nodeORM = context->createShaderNode("Image2DTexture");
            nodeORM->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_OcclusionRoughnessMetallic.png", "NA", "Rec709(D65)"));
            auto nodeNormal = context->createShaderNode("Image2DTexture");
            nodeNormal->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_02_Body_NormalAlpha.png", "NA", "Rec709(D65)"));

            auto ue4Mat = context->createSurfaceMaterial("UE4");
            ue4Mat->set("base color", nodeBaseColor->getPlug(VLRShaderNodePlugType_Spectrum, 0));
            ue4Mat->set("occlusion/roughness/metallic", nodeORM->getPlug(VLRShaderNodePlugType_float3, 0));

            mat = ue4Mat;
            plugNormal = nodeNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);
        }
        else if (strcmp(strValue.C_Str(), "_Base1") == 0) {
            auto nodeBaseColor = context->createShaderNode("Image2DTexture");
            nodeBaseColor->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_BaseColor.png", "Reflectance", "Rec709(D65) sRGB Gamma"));
            auto nodeORM = context->createShaderNode("Image2DTexture");
            nodeORM->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_OcclusionRoughnessMetallic.png", "NA", "Rec709(D65)"));
            auto nodeNormal = context->createShaderNode("Image2DTexture");
            nodeNormal->set("image", loadImage2D(context, pathPrefix + "MeetMat_2_Cameras_03_Base_NormalAlpha.png", "NA", "Rec709(D65)"));

            auto ue4Mat = context->createSurfaceMaterial("UE4");
            ue4Mat->set("base color", nodeBaseColor->getPlug(VLRShaderNodePlugType_Spectrum, 0));
            ue4Mat->set("occlusion/roughness/metallic", nodeORM->getPlug(VLRShaderNodePlugType_float3, 0));

            mat = ue4Mat;
            plugNormal = nodeNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);

            auto nodeTangent = context->createShaderNode("Tangent");
            nodeTangent->set("tangent type", "Radial Y");
            plugTangent = nodeTangent->getPlug(VLRShaderNodePlugType_Vector3D, 0);
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    }, [](const aiMesh* mesh) {
        return MeshAttributeTuple(true);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0.01, 0) * scale<float>(0.25f)));



    construct(context, "resources/sphere/sphere.obj", false, false, &modelNode, [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        SurfaceMaterialRef mat = context->createSurfaceMaterial("SpecularReflection");
        mat->set("coeff", VLRImmediateSpectrum{ "Rec709(D65) sRGB Gamma", 0.999f, 0.999f, 0.999f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.27579f, 0.940922f, 0.574879f }); // Aluminum
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 7.30257f, 6.33458f, 5.16694f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.237698f, 0.734847f, 1.37062f }); // Copper
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.44233f, 2.55751f, 2.23429f });
        mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.12481f, 0.468228f, 1.44476f }); // Gold
        mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.32107f, 2.23761f, 1.69196f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 2.91705f, 2.92092f, 2.53253f }); // Iron
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.06696f, 2.93804f, 2.7429f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.9566f, 1.82777f, 1.46089f }); // Lead
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.49593f, 3.38158f, 3.17737f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 1.99144f, 1.5186f, 1.00058f }); // Mercury
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 5.25161f, 4.6095f, 3.7646f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 2.32528f, 2.06722f, 1.81479f }); // Platinum
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 4.19238f, 3.67941f, 3.06551f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 0.157099f, 0.144013f, 0.134847f }); // Silver
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.82431f, 3.1451f, 2.27711f });
        //mat->set("eta", VLRImmediateSpectrum{ "Rec709(D65)", 2.71866f, 2.50954f, 2.22767f }); // Titanium
        //mat->set("k", VLRImmediateSpectrum{ "Rec709(D65)", 3.79521f, 3.40035f, 3.00114f });

        //SurfaceMaterialRef mat = context->createSurfaceMaterial("SpecularScattering");
        //mat->set("coeff", VLRImmediateSpectrum{ "Rec709(D65)", 0.999f, 0.999f, 0.999f });
        //mat->set("eta ext", VLRImmediateSpectrum{ "Rec709(D65)", 1.00036f, 1.00021f, 1.00071f }); // Air
        //mat->set("eta int", VLRImmediateSpectrum{ "Rec709(D65)", 2.41174f, 2.42343f, 2.44936f }); // Diamond
        ////mat->set("eta int", VLRImmediateSpectrum{ "Rec709(D65)", 1.33161f, 1.33331f, 1.33799f }); // Water
        ////mat->set("eta int", VLRImmediateSpectrum{ "Rec709(D65)", 1.51455f, 1.51816f, 1.52642f }); // Glass BK7

        //SurfaceMaterialRef mats[] = { matA, matB };
        //SurfaceMaterialRef mat = context->createSurfaceMaterial("Multi");

        return SurfaceMaterialAttributeTuple(mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(-2.0f, 0.0f, 2.0f) * scale(1.0f) * translate<float>(0.0f, 1.0f, 0.0f)));



    auto imgEnv = loadImage2D(context, "resources/material_test/Chelsea_Stairs_3k.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = -M_PI / 2;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;
    
    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.0f, 5.0f, 10.0f));
        camera->set("orientation", qRotateY<float>(M_PI) * qRotateX<float>(18 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createGalleryScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

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
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 50.0f, 50.0f, 50.0f });

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 2.0f, 0.0f) * rotateX<float>(M_PI)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    const auto gelleryMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            auto matteMat = context->createSurfaceMaterial("Matte");
            matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.5f, 0.5f, 0.5f });

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };
    construct(context, ASSETS_DIR"gallery/gallery.obj", false, true, &modelNode, createMaterialDefaultFunction);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.5f)));



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;
    shot->environmentRotation = 0.0f;
    
    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(-2.3f, 1.0f, 3.5f));
        camera->set("orientation", qRotateY<float>(0.8 * M_PI) * qRotateX<float>(0));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createHairballScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

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
            auto matLight = context->createSurfaceMaterial("DiffuseEmitter");
            matLight->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", 50.0f, 50.0f, 50.0f });

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            light->addMaterialGroup(matGroup.data(), matGroup.size(), matLight, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    auto lightNode = context->createInternalNode("light", context->createStaticTransform(translate<float>(0.0f, 5.0f, 0.0f) * rotateX<float>(M_PI)));
    lightNode->addChild(light);
    shot->scene->addChild(lightNode);



    construct(context, ASSETS_DIR"hairball/hairball.obj", true, false, &modelNode,
              [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            auto matteMat = context->createSurfaceMaterial("Matte");
            matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.5f, 0.5f, 0.5f });

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    });
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.1f)));



    shot->renderTargetSizeX = 1024;
    shot->renderTargetSizeY = 1024;

    shot->brightnessCoeff = 1.0f;
    shot->environmentRotation = 0.0f;
    
    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.0f, 0.0f, 1.5f));
        camera->set("orientation", qRotateY<float>(M_PI) * qRotateX<float>(0 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createRungholtScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    const auto rungholtMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);
        hpprintf("Material: %s\n", strValue.C_Str());

        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;

        Image2DRef imgDiffuse;
        ShaderNodeRef texDiffuse;
        Image2DRef imgAlpha;
        ShaderNodeRef texAlpha;

        if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
            texDiffuse = context->createShaderNode("Image2DTexture");
            imgDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
            texDiffuse->set("image", imgDiffuse);
            texDiffuse->set("filter", "Nearest");
            mat->set("albedo", texDiffuse->getPlug(VLRShaderNodePlugType_Spectrum, 0));
        }
        else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
            mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", color[0], color[1], color[2] });
        }
        else {
            mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 1.0f, 0.0f, 1.0f });
        }

        //if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
        //    imgAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
        //    texAlpha = context->createShaderNode("Image2DTexture");
        //    texAlpha->set("image", imgAlpha);
        //}

        /*if (imgAlpha) {
            plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_float1, 0);
        }
        else*/ if (imgDiffuse && imgDiffuse->originalHasAlpha()) {
            plugAlpha = texDiffuse->getPlug(VLRShaderNodePlugType_Alpha, 0);
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
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
            auto mat = context->createSurfaceMaterial("Matte");
            mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.05f, 0.025f, 0.025f });

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3
            };
            ground->addMaterialGroup(matGroup.data(), matGroup.size(), mat, ShaderNodePlug(), ShaderNodePlug(), ShaderNodePlug());
        }
    }
    modelNode->addChild(ground);



    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Playa_Sunrise/Playa_Sunrise.exr", "Light Source", "Rec709(D65)");
    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", "Light Source", "Rec709(D65)");
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/Direct_HDR_Capture_of_the_Sun_and_Sky/1400/probe_14-00_latlongmap.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = -0.2 * M_PI;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(10.0f, 5.0f, 0.0f));
        camera->set("orientation", qRotateY<float>(-M_PI / 2 - 0.2 * M_PI) * qRotateX<float>(30 * M_PI / 180));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createPowerplantScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    const auto powerplantMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            auto matteMat = context->createSurfaceMaterial("Matte");
            matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.5f, 0.5f, 0.5f });

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };
    construct(context, ASSETS_DIR"powerplant/powerplant.obj", false, true, &modelNode, createMaterialDefaultFunction);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.0001f)));



    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Barcelona_Rooftops/Barce_Rooftop_C_3k.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(-14.89948f, 1.289585f, -1.764552f));
        camera->set("orientation", Quaternion(-0.089070f, 0.531405f, 0.087888f, 0.837825f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createAmazonBistroExteriorScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    const auto bistroMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

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
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            if (aiMat->Get(AI_MATKEY_COLOR_TRANSPARENT, color, nullptr) != aiReturn_SUCCESS) {
                color[0] = color[1] = color[2] = 1;
            }

            if (color[0] < 1 && color[1] < 1 && color[2] < 1) {
                auto glassMat = context->createSurfaceMaterial("SpecularScattering");
                glassMat->set("coeff", VLRImmediateSpectrum{ "Rec709(D65)", 1 - color[0], 1 - color[1], 1 - color[2] });
                glassMat->set("eta ext", VLRImmediateSpectrum{ "Rec709(D65)", 1.0f, 1.0f, 1.0f });
                glassMat->set("eta int", VLRImmediateSpectrum{ "Rec709(D65)", 1.5f, 1.5f, 1.5f });

                mat = glassMat;
            }
            else {
                auto oldMat = context->createSurfaceMaterial("OldStyle");

                oldMat->set("glossiness", 0.7f);

                Image2DRef imageDiffuse;
                Image2DRef imageSpecular;
                Image2DRef imageNormal;
                Image2DRef imageAlpha;
                ShaderNodeRef texDiffuse;
                ShaderNodeRef texSpecular;
                ShaderNodeRef texNormal;
                ShaderNodeRef texAlpha;

                if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                    imageDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
                    texDiffuse = context->createShaderNode("Image2DTexture");
                    texDiffuse->set("image", imageDiffuse);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
                    imageSpecular = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
                    texSpecular = context->createShaderNode("Image2DTexture");
                    texSpecular->set("image", imageSpecular);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS) {
                    imageNormal = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
                    texNormal = context->createShaderNode("Image2DTexture");
                    texNormal->set("image", imageNormal);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
                    imageAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
                    texAlpha = context->createShaderNode("Image2DTexture");
                    texAlpha->set("image", imageAlpha);
                }

                if (texDiffuse)
                    oldMat->set("diffuse", texDiffuse->getPlug(VLRShaderNodePlugType_Spectrum, 0));

                if (texSpecular)
                    oldMat->set("specular", texSpecular->getPlug(VLRShaderNodePlugType_Spectrum, 0));

                //if (imageSpecular && imageSpecular->originalHasAlpha())
                //    oldMat->set("glossiness", texSpecular->getPlug(VLRShaderNodePlugType_Alpha, 0));

                if (texNormal)
                    plugNormal = texNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);

                if (texAlpha)
                    plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_float1, 0);
                else if (imageDiffuse && imageDiffuse->originalHasAlpha())
                    plugAlpha = texDiffuse->getPlug(VLRShaderNodePlugType_Alpha, 0);

                mat = oldMat;

                if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS) {
                    if (color[0] > 0.0f && color[1] > 0.0f && color[2] > 0.0f) {
                        auto emitter = context->createSurfaceMaterial("DiffuseEmitter");
                        emitter->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", color[0], color[1], color[2] });
                        emitter->set("scale", 30);

                        auto mMat = context->createSurfaceMaterial("Multi");
                        mMat->set("0", oldMat);
                        mMat->set("1", emitter);

                        mat = mMat;
                    }
                }
            }
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };
    const auto grayMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            auto matteMat = context->createSurfaceMaterial("Matte");
            matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.5f, 0.5f, 0.5f });

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };
    construct(context, ASSETS_DIR"Amazon_Bistro/exterior/exterior.obj", false, true, &modelNode, bistroMaterialFunc);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.001f)));



    //Image2DRef imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Barcelona_Rooftops/Barce_Rooftop_C_3k.exr", false);
    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = 0.0f;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;
    
    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(-0.753442f, 0.140257f, -0.056083f));
        camera->set("orientation", Quaternion(-0.009145f, 0.531434f, -0.005825f, 0.847030f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.001f;
        camera->set("sensitivity", 1.0f / (M_PI * lensRadius * lensRadius));
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", lensRadius);
        camera->set("op distance", 0.267f);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createCamera("Equirectangular");

        camera->set("position", Point3D(-1.092485f, 0.640749f, -0.094409f));
        camera->set("orientation", Quaternion(0.109960f, 0.671421f, -0.081981f, 0.812352f));

        float phiAngle = 2.127f;
        float thetaAngle = 1.153f;
        camera->set("sensitivity", 1.0f / (phiAngle * (1 - std::cos(thetaAngle))));
        camera->set("h angle", phiAngle);
        camera->set("v angle", thetaAngle);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(-0.380530f, 0.167073f, -0.309329f));
        camera->set("orientation", Quaternion(0.152768f, 0.422808f, -0.030319f, 0.962553f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", 0.0f);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }
}

void createAmazonBistroInteriorScene(const vlr::ContextRef &context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    const auto bistroMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

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
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            if (aiMat->Get(AI_MATKEY_COLOR_TRANSPARENT, color, nullptr) != aiReturn_SUCCESS) {
                color[0] = color[1] = color[2] = 1;
            }

            if (color[0] < 1 && color[1] < 1 && color[2] < 1) {
                auto glassMat = context->createSurfaceMaterial("SpecularScattering");
                glassMat->set("coeff", VLRImmediateSpectrum{ "Rec709(D65)", 1 - color[0], 1 - color[1], 1 - color[2] });
                glassMat->set("eta ext", VLRImmediateSpectrum{ "Rec709(D65)", 1.0f, 1.0f, 1.0f });
                glassMat->set("eta int", VLRImmediateSpectrum{ "Rec709(D65)", 1.5f, 1.5f, 1.5f });

                mat = glassMat;
            }
            else {
                auto oldMat = context->createSurfaceMaterial("OldStyle");

                oldMat->set("glossiness", 0.7f);

                Image2DRef imageDiffuse;
                Image2DRef imageSpecular;
                Image2DRef imageNormal;
                Image2DRef imageAlpha;
                ShaderNodeRef texDiffuse;
                ShaderNodeRef texSpecular;
                ShaderNodeRef texNormal;
                ShaderNodeRef texAlpha;

                if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
                    imageDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
                    texDiffuse = context->createShaderNode("Image2DTexture");
                    texDiffuse->set("image", imageDiffuse);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_SPECULAR(0), strValue) == aiReturn_SUCCESS) {
                    imageSpecular = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
                    texSpecular = context->createShaderNode("Image2DTexture");
                    texSpecular->set("image", imageSpecular);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS) {
                    imageNormal = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
                    texNormal = context->createShaderNode("Image2DTexture");
                    texNormal->set("image", imageNormal);
                }
                if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
                    imageAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
                    texAlpha = context->createShaderNode("Image2DTexture");
                    texAlpha->set("image", imageAlpha);
                }

                if (texDiffuse)
                    oldMat->set("diffuse", texDiffuse->getPlug(VLRShaderNodePlugType_Spectrum, 0));

                if (texSpecular)
                    oldMat->set("specular", texSpecular->getPlug(VLRShaderNodePlugType_Spectrum, 0));

                //if (imageSpecular && imageSpecular->originalHasAlpha())
                //    oldMat->set("glossiness", texSpecular->getPlug(VLRShaderNodePlugType_Alpha, 0));

                if (texNormal)
                    plugNormal = texNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);

                if (texAlpha)
                    plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_float1, 0);
                else if (imageDiffuse && imageDiffuse->originalHasAlpha())
                    plugAlpha = texDiffuse->getPlug(VLRShaderNodePlugType_Alpha, 0);

                mat = oldMat;

                if (aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, color, nullptr) == aiReturn_SUCCESS) {
                    if (color[0] > 0.0f && color[1] > 0.0f && color[2] > 0.0f) {
                        auto emitter = context->createSurfaceMaterial("DiffuseEmitter");
                        emitter->set("emittance", VLRImmediateSpectrum{ "Rec709(D65)", color[0], color[1], color[2] });
                        emitter->set("scale", 30);

                        auto mMat = context->createSurfaceMaterial("Multi");
                        mMat->set("0", oldMat);
                        mMat->set("1", emitter);

                        mat = mMat;
                    }
                }
            }
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };
    const auto grayMaterialFunc = [](const vlr::ContextRef &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);

        SurfaceMaterialRef mat;
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;
        {
            auto matteMat = context->createSurfaceMaterial("Matte");
            matteMat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 0.5f, 0.5f, 0.5f });

            mat = matteMat;
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };
    construct(context, ASSETS_DIR"Amazon_Bistro/Interior/interior_corrected.obj", false, true, &modelNode, bistroMaterialFunc);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(0.001f)));



    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Barcelona_Rooftops/Barce_Rooftop_C_3k.exr", "Light Source", "Rec709(D65)");
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", "Light Source", "Rec709(D65)");
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/Direct_HDR_Capture_of_the_Sun_and_Sky/1400/probe_14-00_latlongmap.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    matEnv->set("scale", 10);
    //matEnv->set("emittance", RGBSpectrum(0.1f, 0.1f, 0.1f));
    shot->environmentRotation = 160 * M_PI / 180;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(-0.177799f, 0.224542f, -0.070547f));
        camera->set("orientation", Quaternion(0.034520f, 0.748582f, -0.032168f, 0.661360f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.0f;
        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", lensRadius);
        camera->set("op distance", 1.000f);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(0.804731f, 0.146986f, 0.204337f));
        camera->set("orientation", Quaternion(0.101459f, -0.081018f, 0.013512f, 0.991442f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.0001f;
        camera->set("sensitivity", 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", lensRadius);
        camera->set("op distance", 0.036f);

        shot->viewpoints.push_back(camera);
    }
}

void createSanMiguelScene(const vlr::ContextRef& context, Shot* shot) {
    using namespace vlr;

    shot->scene = context->createScene();

    InternalNodeRef modelNode;

    const auto sanMiguelMaterialFunc = [](const vlr::ContextRef& context, const aiMaterial* aiMat, const std::string& pathPrefix) {
        using namespace vlr;

        aiReturn ret;
        (void)ret;
        aiString strValue;
        float color[3];

        aiMat->Get(AI_MATKEY_NAME, strValue);
        hpprintf("Material: %s\n", strValue.C_Str());

        SurfaceMaterialRef mat = context->createSurfaceMaterial("Matte");
        ShaderNodePlug plugNormal;
        ShaderNodePlug plugTangent;
        ShaderNodePlug plugAlpha;

        Image2DRef imgDiffuse;
        ShaderNodeRef texDiffuse;
        Image2DRef imgNormal;
        ShaderNodeRef texNormal;
        Image2DRef imgAlpha;
        ShaderNodeRef texAlpha;

        // Base Color
        if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
            texDiffuse = context->createShaderNode("Image2DTexture");
            imgDiffuse = loadImage2D(context, pathPrefix + strValue.C_Str(), "Reflectance", "Rec709(D65) sRGB Gamma");
            texDiffuse->set("image", imgDiffuse);
            mat->set("albedo", texDiffuse->getPlug(VLRShaderNodePlugType_Spectrum, 0));
        }
        else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
            mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", color[0], color[1], color[2] });
        }
        else {
            mat->set("albedo", VLRImmediateSpectrum{ "Rec709(D65)", 1.0f, 0.0f, 1.0f });
        }

        // Normal
        if (aiMat->Get(AI_MATKEY_TEXTURE_HEIGHT(0), strValue) == aiReturn_SUCCESS) {
            imgNormal = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
            texNormal = context->createShaderNode("Image2DTexture");
            texNormal->set("image", imgNormal);

            std::filesystem::path texPath = strValue.C_Str();
            std::string stem = texPath.stem().string();
            if (stem.find("N_") == std::string::npos)
                texNormal->set("bump type", "Height Map");
            else
                texNormal->set("bump type", "Normal Map (DirectX)");
        }

        // Alpha
        if (aiMat->Get(AI_MATKEY_TEXTURE_OPACITY(0), strValue) == aiReturn_SUCCESS) {
            imgAlpha = loadImage2D(context, pathPrefix + strValue.C_Str(), "NA", "Rec709(D65)");
            texAlpha = context->createShaderNode("Image2DTexture");
            texAlpha->set("image", imgAlpha);
        }

        if (imgNormal)
            plugNormal = texNormal->getPlug(VLRShaderNodePlugType_Normal3D, 0);

        if (imgAlpha) {
            if (std::strcmp(imgAlpha->getOriginalDataFormat(), "Gray8") == 0)
                plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_float1, 0);
            else
                plugAlpha = texAlpha->getPlug(VLRShaderNodePlugType_Alpha, 0);
        }
        else if (imgDiffuse && imgDiffuse->originalHasAlpha()) {
            plugAlpha = texDiffuse->getPlug(VLRShaderNodePlugType_Alpha, 0);
        }

        return SurfaceMaterialAttributeTuple(mat, plugNormal, plugTangent, plugAlpha);
    };

    construct(context, ASSETS_DIR"San_Miguel/san-miguel.obj", false, true, &modelNode, sanMiguelMaterialFunc);
    shot->scene->addChild(modelNode);
    modelNode->setTransform(context->createStaticTransform(translate<float>(0, 0, 0) * scale<float>(1.0f)));



    auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/Direct_HDR_Capture_of_the_Sun_and_Sky/1400/probe_14-00_latlongmap.exr", "Light Source", "Rec709(D65)");
    //auto imgEnv = loadImage2D(context, ASSETS_DIR"IBLs/sIBL_archive/Malibu_Overlook_3k_corrected.exr", "Light Source", "Rec709(D65)");
    auto nodeEnvTex = context->createShaderNode("EnvironmentTexture");
    nodeEnvTex->set("image", imgEnv);
    auto matEnv = context->createSurfaceMaterial("EnvironmentEmitter");
    matEnv->set("emittance", nodeEnvTex->getPlug(VLRShaderNodePlugType_Spectrum, 0));
    //matEnv->set("emittance", RGBSpectrum(0.1f, 0.1f, 0.1f));
    matEnv->set("scale", 100.0f);
    shot->environmentRotation = 250 * M_PI / 180;
    shot->scene->setEnvironment(matEnv, shot->environmentRotation);



    shot->renderTargetSizeX = 1920;
    shot->renderTargetSizeY = 1080;

    shot->brightnessCoeff = 1.0f;

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(6.255f, 1.427f, 6.772f));
        camera->set("orientation", Quaternion(0.009f, 0.865f, -0.009f, 0.502f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.0f;
        camera->set("sensitivity", lensRadius > 0.0f ? 1.0f / (M_PI * lensRadius * lensRadius) : 1.0f);
        camera->set("fovy", 40 * M_PI / 180);
        camera->set("lens radius", lensRadius);
        camera->set("op distance", 1.0f);

        shot->viewpoints.push_back(camera);
    }

    {
        auto camera = context->createCamera("Perspective");

        camera->set("position", Point3D(25.535f, 1.531f, 1.669f));
        camera->set("orientation", Quaternion(-0.063f, 0.557f, -0.054f, -0.827f));

        camera->set("aspect", (float)shot->renderTargetSizeX / shot->renderTargetSizeY);

        float lensRadius = 0.01f;
        camera->set("sensitivity", lensRadius > 0.0f ? 1.0f / (M_PI * lensRadius * lensRadius) : 1.0f);
        camera->set("fovy", 65 * M_PI / 180);
        camera->set("lens radius", lensRadius);
        camera->set("op distance", 3.779f);

        shot->viewpoints.push_back(camera);
    }
}

void createScene(const vlr::ContextRef &context, Shot* shot) {
    //createCornellBoxScene(context, shot);
    //createMaterialTestScene(context, shot);
    //createAnisotropyScene(context, shot);
    //createWhiteFurnaceTestScene(context, shot);
    //createColorCheckerScene(context, shot);
    //createColorInterpolationTestScene(context, shot);
    createSubstanceManScene(context, shot);
    //createGalleryScene(context, shot);
    //createHairballScene(context, shot);
    //createRungholtScene(context, shot);
    //createPowerplantScene(context, shot);
    //createAmazonBistroExteriorScene(context, shot);
    //createAmazonBistroInteriorScene(context, shot);
    //createSanMiguelScene(context, shot);
    context->setScene(shot->scene);
}
