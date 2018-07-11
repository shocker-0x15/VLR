#include "scene_private.h"

namespace VLR {
    // ----------------------------------------------------------------
    // Shallow Hierarchy

    void SHGroup::addChild(SHTransform* transform) {
        m_transforms.insert(transform);
        transform->setParent(this);
        m_optixGroup->addChild(transform->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGroup::addChild(SHGeometryGroup* geomGroup) {
        m_geometryGroups.insert(geomGroup);
        m_optixGroup->addChild(geomGroup->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGroup::removeChild(SHTransform* transform) {
        m_optixGroup->removeChild(transform->getOptiXObject());
        transform->setParent(nullptr);
        m_transforms.erase(transform);
        m_optixAcceleration->markDirty();
    }

    void SHGroup::removeChild(SHGeometryGroup* geomGroup) {
        m_optixGroup->removeChild(geomGroup->getOptiXObject());
        m_geometryGroups.erase(geomGroup);
        m_optixAcceleration->markDirty();
    }

    void SHGroup::childUpdateEvent(const SHTransform* transform) {
        m_optixAcceleration->markDirty();
    }

    void SHGroup::childUpdateEvent(const SHGeometryGroup* geomGroup) {
        m_optixAcceleration->markDirty();
    }



    void SHTransform::resolveTransform() {
        int32_t stackIdx = 0;
        const SHTransform* stack[5];
        std::fill_n(stack, lengthof(stack), nullptr);
        const SHTransform* nextSHTr = (m_childIsTransform && m_childTransform) ? m_childTransform : nullptr;
        while (nextSHTr) {
            stack[stackIdx++] = nextSHTr;
            nextSHTr = (nextSHTr->m_childIsTransform && nextSHTr->m_childTransform) ? nextSHTr->m_childTransform : nullptr;
        }

        StaticTransform res;
        std::string concatenatedName = "";
        --stackIdx;
        while (stackIdx >= 0) {
            const SHTransform* shtr = stack[stackIdx--];
            res = shtr->m_transform * res;
            concatenatedName = "-" + shtr->getName() + concatenatedName;
        }
        res = m_transform * res;
        concatenatedName = m_name + concatenatedName;

        float mat[16], invMat[16];
        res.getArrays(mat, invMat);
        m_optixTransform->setMatrix(false, mat, invMat);

        // JP: 親に変形情報が更新されたことを通知する。
        // EN: notify the parent that transform has updated.
        if (m_parent)
            m_parent->childUpdateEvent(this);

        //if (true/*m_parent*/) {
        //    VLRDebugPrintf("%s:\n", concatenatedName.c_str());
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
        //    VLRDebugPrintf("%g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
        //    VLRDebugPrintf("\n");
        //}
    }

    void SHTransform::setTransform(const StaticTransform &transform) {
        m_transform = transform;
        resolveTransform();
    }

    void SHTransform::update() {
        resolveTransform();
    }

    void SHTransform::addChild(SHGeometryGroup* geomGroup) {
        VLRAssert(!m_childIsTransform, "Transform which doesn't have a child transform can have a geometry group as a child.");
        m_childGeometryGroup = geomGroup;
    }



    void SHGeometryGroup::addGeometryInstance(SHGeometryInstance* instance) {
        m_instances.push_back(instance);
        instance->setParent(this);
        m_optixGeometryGroup->addChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGeometryGroup::childUpdateEvent(const SHGeometryInstance* instance) {
        m_optixAcceleration->markDirty();
    }

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    void ParentNode::addChild(const InternalNodeRef &child) {
        m_children.insert(child);
        child->addParent(this);
    }

    void ParentNode::addChild(const SurfaceNodeRef &child) {
        m_children.insert(child);
        child->addParent(this);
    }

    void ParentNode::removeChild(const InternalNodeRef &child) {
        m_children.erase(child);
        child->removeParent(this);
    }

    void ParentNode::removeChild(const SurfaceNodeRef &child) {
        m_children.erase(child);
        child->removeParent(this);
    }

    void ParentNode::setTransform(const TransformRef &localToWorld) {
        m_localToWorld = localToWorld;

        // JP: 管理中のSHTransformを更新する。
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it) {
            if (m_localToWorld->isStatic()) {
                StaticTransform* tr = (StaticTransform*)m_localToWorld.get();
                SHTransform* shtr = it->second;
                shtr->setTransform(*tr);
            }
            else {
                VLRAssert_NotImplemented();
            }
        }
    }



    void InternalNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) {
        switch (eventType) {
        case UpdateEvent::Added: {
            // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
            //     子のSHTransformをキーとして辞書に保存する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_localToWorld->isStatic()) {
                    StaticTransform* tr = (StaticTransform*)m_localToWorld.get();
                    SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                    m_shTransforms[*it] = shtr;
                    delta.insert(shtr);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }

            // JP: 親に自分が保持するSHTransformが増えたことを通知(増分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta);
            }

            break;
        }
        case UpdateEvent::Removed: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを削除。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shTransforms.erase(*it);
                delta.insert(shtr);
            }

            // JP: 親に自分が保持するSHTransformが減ったことを通知(減分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta);
            }

            for (auto it = delta.cbegin(); it != delta.cend(); ++it)
                delete *it;

            break;
        }
        case UpdateEvent::Updated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
                delta.insert(shtr);
            }

            // JP: 親に自分が保持するSHTransformが更新されたことを通知(更新分を通知)。
            for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                auto parent = *it;
                parent->childUpdateEvent(eventType, delta);
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    void InternalNode::setTransform(const TransformRef &localToWorld) {
        ParentNode::setTransform(localToWorld);

        // JP: 親ノードに変形情報が更新されたことを通知する。
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(UpdateEvent::Updated, delta);
        }
    }

    void InternalNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);

        // JP: 追加した親に対して変形情報の追加を行わせる。
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);
        parent->childUpdateEvent(UpdateEvent::Added, delta);
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 削除した親に対して変形情報の削除を行わせる。
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);
        parent->childUpdateEvent(UpdateEvent::Removed, delta);
    }



    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) {
        switch (eventType) {
        case UpdateEvent::Added: {
            // JP: 自分自身のTransformと子InternalNodeが持つSHTransformを繋げたSHTransformを生成。
            //     子のSHTransformをキーとして辞書に保存する。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                if (m_localToWorld->isStatic()) {
                    StaticTransform* tr = (StaticTransform*)m_localToWorld.get();
                    SHTransform* shtr = new SHTransform(m_name, m_context, *tr, *it);
                    m_shTransforms[*it] = shtr;
                    delta.insert(shtr);
                }
                else {
                    VLRAssert_NotImplemented();
                }
            }

            // JP: SHGroupにもSHTransformを追加する。
            for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
                SHTransform* shtr = *it;
                m_shGroup.addChild(shtr);
            }

            break;
        }
        case UpdateEvent::Removed: {
            // JP: 子InternalNodeが持つSHTransformがつながっているSHTransformを削除。
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shTransforms.erase(*it);
                delta.insert(shtr);
            }

            // JP: SHGroupからもSHTransformを削除する。
            for (auto it = delta.cbegin(); it != delta.cend(); ++it) {
                SHTransform* shtr = *it;
                m_shGroup.removeChild(shtr);
            }

            for (auto it = delta.cbegin(); it != delta.cend(); ++it)
                delete *it;

            break;
        }
        case UpdateEvent::Updated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }



    void SurfaceNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);
    }

    void SurfaceNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);
    }


    
    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        optix::Context &optixContext = m_context.getOptiXContext();
        m_optixVertexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_vertices.size());
        m_optixVertexBuffer->setElementSize(sizeof(Shared::Vertex));
        auto dstVertices = (Shared::Vertex*)m_optixVertexBuffer->map();
        {
            static_assert(sizeof(Vertex) == sizeof(Shared::Vertex), "These two types must match in size (and structure).");
            std::copy_n((Shared::Vertex*)m_vertices.data(), m_vertices.size(), dstVertices);
        }
        m_optixVertexBuffer->unmap();
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterialRef &material) {
        VLRAssert_NotImplemented();
    }



    const size_t sizesOfDataFormats[(uint32_t)DataFormat::Num] = {
        sizeof(RGB8x3), 
        sizeof(RGB_8x4), 
        sizeof(RGBA8x4), 
        //sizeof(RGBA16Fx4), 
        sizeof(Gray8),
    };

    DataFormat Image2D::getInternalFormat(DataFormat inputFormat) {
        switch (inputFormat) {
        case DataFormat::RGB8x3:
            return DataFormat::RGBA8x4;
        case DataFormat::RGB_8x4:
            return DataFormat::RGBA8x4;
        case DataFormat::RGBA8x4:
            return DataFormat::RGBA8x4;
        case DataFormat::Gray8:
            return DataFormat::Gray8;
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }
        return DataFormat::RGBA8x4;
    }

    LinearImage2D::LinearImage2D(const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat) :
    Image2D(width, height, Image2D::getInternalFormat(dataFormat)) {
        uint32_t inputStride = (uint32_t)sizesOfDataFormats[(uint32_t)dataFormat];

        std::function<void(const uint8_t*, uint32_t, uint32_t, uint8_t*)> funcConvertLine;
        switch (dataFormat) {
        case DataFormat::RGB8x3:
            funcConvertLine = [](const uint8_t* linearData, uint32_t width, uint32_t y, uint8_t* lineMemory) {
                auto* srcHeader = (const RGB8x3*)linearData + width * y;
                auto* dstHeader = (RGBA8x4*)lineMemory;
                for (int x = 0; x < width; ++x) {
                    const RGB8x3 &src = *(srcHeader + x);
                    RGBA8x4 &dst = *(dstHeader + x);
                    dst.r = src.r;
                    dst.g = src.g;
                    dst.b = src.b;
                    dst.a = 255;
                }
            };
            break;
        case DataFormat::RGB_8x4:
            funcConvertLine = [](const uint8_t* linearData, uint32_t width, uint32_t y, uint8_t* lineMemory) {
                auto* srcHeader = (const RGB_8x4*)linearData + width * y;
                auto* dstHeader = (RGBA8x4*)lineMemory;
                std::copy_n((const RGBA8x4*)srcHeader, width, dstHeader);
                for (int x = 0; x < width; ++x) {
                    RGBA8x4 &dst = *(dstHeader + x);
                    dst.a = 255;
                }
            };
            break;
        case DataFormat::RGBA8x4:
            funcConvertLine = [](const uint8_t* linearData, uint32_t width, uint32_t y, uint8_t* lineMemory) {
                auto* srcHeader = (const RGBA8x4*)linearData + width * y;
                std::copy_n(srcHeader, width, (RGBA8x4*)lineMemory);
            };
            break;
        case DataFormat::Gray8:
            funcConvertLine = [](const uint8_t* linearData, uint32_t width, uint32_t y, uint8_t* lineMemory) {
                auto* srcHeader = (const Gray8*)linearData + width * y;
                std::copy_n(srcHeader, width, (Gray8*)lineMemory);
            };
            break;
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }

        m_data.resize(getStride() * getWidth() * getHeight());
        std::vector<uint8_t> lineMemory;
        lineMemory.resize(width * 8);
        uint32_t lineDataSize = getStride() * getWidth();
        for (uint32_t y = 0; y < getHeight(); ++y) {
            funcConvertLine(linearData, width, y, lineMemory.data());
            std::copy(lineMemory.cbegin(), lineMemory.cbegin() + lineDataSize, m_data.begin() + y * lineDataSize);
        }
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
    
    VLR_API void test() {
        try {
            Context context;

            optix::Context optixContext = context.getOptiXContext();
            optixContext->setRayTypeCount(3);
            optixContext->setEntryPointCount(1);

            {
                std::string triangle_intersection_ptx = readTxtFile("ptxes/triangle_intersection.ptx");

                optix::Program progIntersectTriangle = optixContext->createProgramFromPTXString(triangle_intersection_ptx, "VLR::intersectTriangle");
                optix::Program progCalcBBoxForTriangle = optixContext->createProgramFromPTXString(triangle_intersection_ptx, "VLR::calcBBoxForTriangle");

                optix::Program progDecodeHitPoint = optixContext->createProgramFromPTXString(triangle_intersection_ptx, "VLR::decodeHitPoint");
                optix::Program progDecodeTexCoord = optixContext->createProgramFromPTXString(triangle_intersection_ptx, "VLR::decodeTexCoord");

                optix::Program progSampleTriangleMesh = optixContext->createProgramFromPTXString(triangle_intersection_ptx, "VLR::sampleTriangleMesh");
            }

            {
                std::string materials_ptx = readTxtFile("ptxes/materials.ptx");

                optix::Program progNormalAlphaModifier_fetchAlpha = optixContext->createProgramFromPTXString(materials_ptx, "VLR::NormalAlphaModifier_fetchAlpha");
                optix::Program progNormalAlphaModifier_fetchNormal = optixContext->createProgramFromPTXString(materials_ptx, "VLR::NormalAlphaModifier_fetchNormal");

                optix::Program progLambertianBRDF_matches = optixContext->createProgramFromPTXString(materials_ptx, "VLR::LambertianBRDF_matches");
                optix::Program progLambertianBRDF_sampleBSDFInternal = optixContext->createProgramFromPTXString(materials_ptx, "VLR::LambertianBRDF_sampleBSDFInternal");
                optix::Program progLambertianBRDF_evaluateBSDFInternal = optixContext->createProgramFromPTXString(materials_ptx, "VLR::LambertianBRDF_evaluateBSDFInternal");
                optix::Program progLambertianBRDF_evaluateBSDF_PDFInternal = optixContext->createProgramFromPTXString(materials_ptx, "VLR::LambertianBRDF_evaluateBSDF_PDFInternal");
                optix::Program progDiffuseEDF_evaluateEDFInternal = optixContext->createProgramFromPTXString(materials_ptx, "VLR::DiffuseEDF_evaluateEDFInternal");
            }
            
            {
                std::string path_tracing_ptx = readTxtFile("ptxes/path_tracing.ptx");
                optix::Program progStochasticAlphaAnyHit = optixContext->createProgramFromPTXString(path_tracing_ptx, "VLR::stochasticAlphaAnyHit");
                optix::Program progAlphaAnyHit = optixContext->createProgramFromPTXString(path_tracing_ptx, "VLR::alphaAnyHit");
                optix::Program progPathTracingIteration = optixContext->createProgramFromPTXString(path_tracing_ptx, "VLR::pathTracingIteration");
                optix::Program progPathTracing = optixContext->createProgramFromPTXString(path_tracing_ptx, "VLR::pathTracing");
            }


            std::shared_ptr<RootNode> rootNode = createShared<RootNode>(context, createShared<StaticTransform>(translate(0.0f, 3.0f, 0.0f)));

            InternalNodeRef nodeA = createShared<InternalNode>("A", context, createShared<StaticTransform>(translate(-5.0f, 0.0f, 0.0f)));
            InternalNodeRef nodeB = createShared<InternalNode>("B", context, createShared<StaticTransform>(translate(5.0f, 0.0f, 0.0f)));

            rootNode->addChild(nodeA);
            rootNode->addChild(nodeB);

            InternalNodeRef nodeC = createShared<InternalNode>("C", context, createShared<StaticTransform>(translate(0.0f, 5.0f, 0.0f)));

            nodeA->addChild(nodeC);
            nodeB->addChild(nodeC);

            InternalNodeRef nodeD = createShared<InternalNode>("D", context, createShared<StaticTransform>(translate(0.0f, 0.0f, -5.0f)));
            InternalNodeRef nodeE = createShared<InternalNode>("E", context, createShared<StaticTransform>(translate(0.0f, 0.0f, 5.0f)));

            nodeC->addChild(nodeD);
            nodeC->addChild(nodeE);

            rootNode->setTransform(createShared<StaticTransform>(translate(0.0f, 5.0f, 0.0f)));
            nodeC->setTransform(createShared<StaticTransform>(translate(0.0f, -10.0f, 0.0f)));

            rootNode->removeChild(nodeB);

            printf("");
        }
        catch (optix::Exception ex) {
            VLRDebugPrintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
        }
    }
}
