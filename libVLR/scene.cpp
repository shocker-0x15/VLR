#include "scene_private.h"

// DELETE ME
#define STB_IMAGE_IMPLEMENTATION
#include "../HostProgram/ext/include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "../HostProgram/ext/include/stb_image_write.h"
#include <random>

namespace VLR {
    static std::string readTxtFile(const std::string& filepath) {
        std::ifstream ifs;
        ifs.open(filepath, std::ios::in);
        if (ifs.fail())
            return "";

        std::stringstream sstream;
        sstream << ifs.rdbuf();

        return std::string(sstream.str());
    };



    uint32_t Context::NextID = 0;
    
    Context::Context() {
        m_ID = getInstanceID();

        m_optixContext = optix::Context::create();

        m_optixContext->setRayTypeCount(Shared::RayType::NumTypes);

        std::string ptx = readTxtFile("resources/ptxes/path_tracing.ptx");

        m_optixCallableProgramNullFetchAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::Null_NormalAlphaModifier_fetchAlpha");
        m_optixCallableProgramNullFetchNormal = m_optixContext->createProgramFromPTXString(ptx, "VLR::Null_NormalAlphaModifier_fetchNormal");
        m_optixCallableProgramFetchAlpha = m_optixContext->createProgramFromPTXString(ptx, "VLR::NormalAlphaModifier_fetchAlpha");
        m_optixCallableProgramFetchNormal = m_optixContext->createProgramFromPTXString(ptx, "VLR::NormalAlphaModifier_fetchNormal");

        m_optixProgramStochasticAlphaAnyHit = m_optixContext->createProgramFromPTXString(ptx, "VLR::stochasticAlphaAnyHit");
        m_optixProgramAlphaAnyHit = m_optixContext->createProgramFromPTXString(ptx, "VLR::alphaAnyHit");
        m_optixProgramPathTracingIteration = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingIteration");

        m_optixProgramPathTracingMiss = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracingMiss");
        m_optixProgramPathTracing = m_optixContext->createProgramFromPTXString(ptx, "VLR::pathTracing");
        m_optixProgramException = m_optixContext->createProgramFromPTXString(ptx, "VLR::exception");

        m_optixContext->setEntryPointCount(1);
        m_optixContext->setRayGenerationProgram(0, m_optixProgramPathTracing);
        m_optixContext->setMissProgram(0, m_optixProgramPathTracingMiss);
        m_optixContext->setExceptionProgram(0, m_optixProgramException);
    }



    // ----------------------------------------------------------------
    // Shallow Hierarchy

    void SHGroup::addChild(SHTransform* transform) {
        TransformStatus status;
        status.hasGeometryDescendant = transform->hasGeometryDescendant();
        m_transforms[transform] = status;
        if (status.hasGeometryDescendant) {
            m_optixGroup->addChild(transform->getOptiXObject());
            m_optixAcceleration->markDirty();
            ++m_numValidTransforms;
        }
    }

    void SHGroup::addChild(SHGeometryGroup* geomGroup) {
        m_geometryGroups.insert(geomGroup);
        m_optixGroup->addChild(geomGroup->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGroup::removeChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        const TransformStatus status = m_transforms.at(transform);
        m_transforms.erase(transform);
        if (status.hasGeometryDescendant) {
            m_optixGroup->removeChild(transform->getOptiXObject());
            m_optixAcceleration->markDirty();
            --m_numValidTransforms;
        }
    }

    void SHGroup::removeChild(SHGeometryGroup* geomGroup) {
        m_optixGroup->removeChild(geomGroup->getOptiXObject());
        m_geometryGroups.erase(geomGroup);
        m_optixAcceleration->markDirty();
    }

    void SHGroup::updateChild(SHTransform* transform) {
        VLRAssert(m_transforms.count(transform), "transform 0x%p is not a child.", transform);
        TransformStatus &status = m_transforms.at(transform);
        SHGeometryGroup* descendant;
        optix::Transform &optixTransform = transform->getOptiXObject();
        if (status.hasGeometryDescendant) {
            if (!transform->hasGeometryDescendant()) {
                m_optixGroup->removeChild(optixTransform);
                m_optixAcceleration->markDirty();
                status.hasGeometryDescendant = false;
                --m_numValidTransforms;
            }
        }
        else {
            if (transform->hasGeometryDescendant(&descendant)) {
                optixTransform->setChild(descendant->getOptiXObject());

                m_optixGroup->addChild(optixTransform);
                m_optixAcceleration->markDirty();
                status.hasGeometryDescendant = true;
                ++m_numValidTransforms;
            }
        }
    }

    void SHGroup::printOptiXHierarchy() {
        std::stack<RTobject> stackRTObjects;
        std::stack<RTobjecttype> stackRTObjectTypes;

        std::set<RTgroup> groupList;
        std::set<RTtransform> transformList;
        std::set<RTgeometrygroup> geometryGroupList;
        std::set<RTgeometryinstance> geometryInstanceList;

        stackRTObjects.push(m_optixGroup.get()->get());
        stackRTObjectTypes.push(RT_OBJECTTYPE_GROUP);
        while (!stackRTObjects.empty()) {
            RTobject object = stackRTObjects.top();
            RTobjecttype objType = stackRTObjectTypes.top();
            stackRTObjects.pop();
            stackRTObjectTypes.pop();

            VLRDebugPrintf("0x%p: ", object);

            switch (objType) {
            case RT_OBJECTTYPE_GROUP: {
                auto group = (RTgroup)object;
                VLRDebugPrintf("Group\n");

                groupList.insert(group);

                uint32_t numChildren;
                rtGroupGetChildCount(group, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTobject childObject = nullptr;
                    RTobjecttype childObjType;
                    rtGroupGetChild(group, i, &childObject);
                    rtGroupGetChildType(group, i, &childObjType);

                    VLRDebugPrintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(childObjType);
                }

                break;
            }
            case RT_OBJECTTYPE_TRANSFORM: {
                auto transform = (RTtransform)object;
                VLRDebugPrintf("Transform\n");

                transformList.insert(transform);

                RTobject childObject = nullptr;
                RTobjecttype childObjType;
                rtTransformGetChild(transform, &childObject);
                rtTransformGetChildType(transform, &childObjType);

                VLRDebugPrintf("- 0x%p\n", childObject);

                stackRTObjects.push(childObject);
                stackRTObjectTypes.push(childObjType);

                break;
            }
            case RT_OBJECTTYPE_SELECTOR: {
                VLRAssert_NotImplemented();
                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_GROUP: {
                auto geometryGroup = (RTgeometrygroup)object;
                VLRDebugPrintf("GeometryGroup\n");

                geometryGroupList.insert(geometryGroup);

                uint32_t numChildren;
                rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
                for (int i = numChildren - 1; i >= 0; --i) {
                    RTgeometryinstance childObject = nullptr;
                    rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                    VLRDebugPrintf("- %u: 0x%p\n", i, childObject);

                    stackRTObjects.push(childObject);
                    stackRTObjectTypes.push(RT_OBJECTTYPE_GEOMETRY_INSTANCE);
                }

                break;
            }
            case RT_OBJECTTYPE_GEOMETRY_INSTANCE: {
                auto geometryInstance = (RTgeometryinstance)object;
                VLRDebugPrintf("GeometryInstance\n");

                geometryInstanceList.insert(geometryInstance);

                break;
            }
            default:
                VLRDebugPrintf("\n");
                VLRAssert_ShouldNotBeCalled();
                break;
            }

            VLRDebugPrintf("\n");
        }



        VLRDebugPrintf("Groups:\n");
        for (auto group : groupList) {
            VLRDebugPrintf("  0x%p:\n", group);
            uint32_t numChildren;
            rtGroupGetChildCount(group, &numChildren);
            RTacceleration acceleration;
            rtGroupGetAcceleration(group, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            VLRDebugPrintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTobject childObject = nullptr;
                rtGroupGetChild(group, i, &childObject);

                VLRDebugPrintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        VLRDebugPrintf("Transforms:\n");
        for (auto transform : transformList) {
            VLRDebugPrintf("  0x%p:\n", transform);
            RTobject childObject = nullptr;
            rtTransformGetChild(transform, &childObject);
            float mat[16];
            float invMat[16];
            rtTransformGetMatrix(transform, true, mat, invMat);
            VLRDebugPrintf("    Matrix\n");
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[0], mat[4], mat[8], mat[12]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[1], mat[5], mat[9], mat[13]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[2], mat[6], mat[10], mat[14]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", mat[3], mat[7], mat[11], mat[15]);
            VLRDebugPrintf("    Inverse Matrix\n");
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[0], invMat[4], invMat[8], invMat[12]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[1], invMat[5], invMat[9], invMat[13]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[2], invMat[6], invMat[10], invMat[14]);
            VLRDebugPrintf("      %g, %g, %g, %g\n", invMat[3], invMat[7], invMat[11], invMat[15]);

            VLRDebugPrintf("  - 0x%p\n", childObject);
        }

        VLRDebugPrintf("GeometryGroups:\n");
        for (auto geometryGroup : geometryGroupList) {
            VLRDebugPrintf("  0x%p:\n", geometryGroup);
            uint32_t numChildren;
            rtGeometryGroupGetChildCount(geometryGroup, &numChildren);
            RTacceleration acceleration;
            rtGeometryGroupGetAcceleration(geometryGroup, &acceleration);
            int32_t isDirty = 0;
            rtAccelerationIsDirty(acceleration, &isDirty);
            VLRDebugPrintf("  Status: %s\n", isDirty ? "dirty" : "");
            for (int i = 0; i < numChildren; ++i) {
                RTgeometryinstance childObject = nullptr;
                rtGeometryGroupGetChild(geometryGroup, i, &childObject);

                VLRDebugPrintf("  - %u: 0x%p\n", i, childObject);
            }
        }

        VLRDebugPrintf("GeometryInstances:\n");
        for (auto geometryInstance : geometryInstanceList) {
            VLRDebugPrintf("  0x%p:\n", geometryInstance);
        }
    }



    void SHTransform::resolveTransform() {
        int32_t stackIdx = 0;
        const SHTransform* stack[5];
        std::fill_n(stack, lengthof(stack), nullptr);
        const SHTransform* nextSHTr = m_childIsTransform ? m_childTransform : nullptr;
        while (nextSHTr) {
            stack[stackIdx++] = nextSHTr;
            nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
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
        m_optixTransform->setMatrix(true, mat, invMat);

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

    void SHTransform::setChild(SHGeometryGroup* geomGroup) {
        VLRAssert(!m_childIsTransform, "Transform which doesn't have a child transform can have a geometry group as a child.");
        m_childGeometryGroup = geomGroup;
    }

    bool SHTransform::hasGeometryDescendant(SHGeometryGroup** descendant) const {
        if (descendant)
            *descendant = nullptr;

        const SHTransform* nextSHTr = this;
        while (nextSHTr) {
            if (!nextSHTr->m_childIsTransform && nextSHTr->m_childGeometryGroup != nullptr) {
                if (descendant)
                    *descendant = nextSHTr->m_childGeometryGroup;
                return true;
            }
            else {
                nextSHTr = nextSHTr->m_childIsTransform ? nextSHTr->m_childTransform : nullptr;
            }
        }

        return false;
    }



    void SHGeometryGroup::addGeometryInstance(SHGeometryInstance* instance) {
        m_instances.insert(instance);
        m_optixGeometryGroup->addChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    void SHGeometryGroup::removeGeometryInstance(SHGeometryInstance* instance) {
        m_instances.erase(instance);
        m_optixGeometryGroup->removeChild(instance->getOptiXObject());
        m_optixAcceleration->markDirty();
    }

    // END: Shallow Hierarchy
    // ----------------------------------------------------------------



    ParentNode::ParentNode(const std::string &name, Context &context, const TransformRef &localToWorld) : 
        Node(name, context), m_localToWorld(localToWorld), m_shGeomGroup(context) {
        // JP: 自分自身のTransformを持ったSHTransformを生成。
        // EN: 
        if (m_localToWorld->isStatic()) {
            StaticTransform* tr = (StaticTransform*)m_localToWorld.get();
            m_shTransforms[nullptr] = new SHTransform(name, m_context, *tr, nullptr);
        }
        else {
            VLRAssert_NotImplemented();
        }
    }

    ParentNode::~ParentNode() {
        for (auto it = m_shTransforms.crbegin(); it != m_shTransforms.crend(); ++it)
            delete it->second;
        m_shTransforms.clear();
    }
    
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
        case UpdateEvent::TransformAdded: {
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
        case UpdateEvent::TransformRemoved: {
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
        case UpdateEvent::TransformUpdated: {
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
        case UpdateEvent::GeometryAdded:
        case UpdateEvent::GeometryRemoved: {
            std::set<SHTransform*> delta;
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
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

    void InternalNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) {
        switch (eventType) {
        case UpdateEvent::GeometryAdded: {
            // JP: 
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.addGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() > 0) {
                SHTransform* selfTransform = m_shTransforms.at(nullptr);
                selfTransform->setChild(&m_shGeomGroup);
                
                std::set<SHTransform*> delta;
                delta.insert(selfTransform);
                for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                    ParentNode* parent = *it;
                    parent->childUpdateEvent(eventType, delta);
                }
            }

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: 
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.removeGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() == 0) {
                SHTransform* selfTransform = m_shTransforms.at(nullptr);
                selfTransform->setChild(nullptr);

                std::set<SHTransform*> delta;
                delta.insert(selfTransform);
                for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
                    ParentNode* parent = *it;
                    parent->childUpdateEvent(eventType, delta);
                }
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    InternalNode::InternalNode(const std::string &name, Context &context, const TransformRef &localToWorld) :
        ParentNode(name, context, localToWorld) {
    }

    void InternalNode::setTransform(const TransformRef &localToWorld) {
        ParentNode::setTransform(localToWorld);

        // JP: 親に変形情報が更新されたことを通知する。
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(UpdateEvent::TransformUpdated, delta);
        }
    }

    void InternalNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);

        // JP: 追加した親に対して変形情報の追加を行わせる。
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);
        parent->childUpdateEvent(UpdateEvent::TransformAdded, delta);
    }

    void InternalNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);

        // JP: 削除した親に対して変形情報の削除を行わせる。
        std::set<SHTransform*> delta;
        for (auto it = m_shTransforms.cbegin(); it != m_shTransforms.cend(); ++it)
            delta.insert(it->second);
        parent->childUpdateEvent(UpdateEvent::TransformRemoved, delta);
    }



    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHTransform*>& childDelta) {
        switch (eventType) {
        case UpdateEvent::TransformAdded: {
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
        case UpdateEvent::TransformRemoved: {
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
        case UpdateEvent::TransformUpdated: {
            // JP: 子InternalNodeが持つSHTransformが繋がっているSHTransformを更新する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                shtr->update();
            }

            break;
        }
        case UpdateEvent::GeometryAdded:
        case UpdateEvent::GeometryRemoved: {
            // JP: SHGroupに対してSHTransformの末尾のジオメトリ状態に変化があったことを通知する。
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it) {
                SHTransform* shtr = m_shTransforms.at(*it);
                m_shGroup.updateChild(shtr);
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    void RootNode::childUpdateEvent(UpdateEvent eventType, const std::set<SHGeometryInstance*> &childDelta) {
        switch (eventType) {
        case UpdateEvent::GeometryAdded: {
            // JP: 
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.addGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() > 0) {
                SHTransform* selfTransform = m_shTransforms.at(nullptr);
                selfTransform->setChild(&m_shGeomGroup);
                m_shGroup.updateChild(selfTransform);
            }

            break;
        }
        case UpdateEvent::GeometryRemoved: {
            // JP: 
            for (auto it = childDelta.cbegin(); it != childDelta.cend(); ++it)
                m_shGeomGroup.removeGeometryInstance(*it);

            if (m_shGeomGroup.getNumInstances() == 0) {
                SHTransform* selfTransform = m_shTransforms.at(nullptr);
                selfTransform->setChild(nullptr);
                m_shGroup.updateChild(selfTransform);
            }

            break;
        }
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }

    RootNode::RootNode(Context &context, const TransformRef &localToWorld) :
        ParentNode("Root", context, localToWorld), m_shGroup(context) {
        SHTransform* shtr = m_shTransforms[0];
        m_shGroup.addChild(shtr);
    }



    // static
    void SurfaceNode::init(Context &context) {
        TriangleMeshSurfaceNode::init(context);
    }
    
    void SurfaceNode::addParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.insert(parent);
    }

    void SurfaceNode::removeParent(ParentNode* parent) {
        VLRAssert(parent != nullptr, "parent must be not null.");
        m_parents.erase(parent);
    }


    
    std::map<uint32_t, TriangleMeshSurfaceNode::OptiXProgramSet> TriangleMeshSurfaceNode::OptiXProgramSets;
    
    // static
    void TriangleMeshSurfaceNode::init(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/triangle_intersection.ptx");

        OptiXProgramSet programSet;

        optix::Context &optixContext = context.getOptiXContext();

        programSet.programIntersectTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::intersectTriangle");
        programSet.programCalcBBoxForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::calcBBoxForTriangle");

        programSet.callableProgramDecodeHitPointForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::decodeHitPointForTriangle");
        programSet.callableProgramDecodeTexCoordForTriangle = optixContext->createProgramFromPTXString(ptx, "VLR::decodeTexCoordForTriangle");

        programSet.callableProgramSampleTriangleMesh = optixContext->createProgramFromPTXString(ptx, "VLR::sampleTriangleMesh");

        OptiXProgramSets[context.getID()] = programSet;
    }

    TriangleMeshSurfaceNode::TriangleMeshSurfaceNode(const std::string &name, Context &context) : SurfaceNode(name, context) {
    }

    TriangleMeshSurfaceNode::~TriangleMeshSurfaceNode() {
        for (auto it = m_shGeometryInstances.crbegin(); it != m_shGeometryInstances.crend(); ++it)
            delete *it;
        m_shGeometryInstances.clear();
    }

    void TriangleMeshSurfaceNode::addParent(ParentNode* parent) {
        SurfaceNode::addParent(parent);

        // JP: 追加した親に対してジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
    }

    void TriangleMeshSurfaceNode::removeParent(ParentNode* parent) {
        SurfaceNode::removeParent(parent);

        // JP: 削除した親に対してジオメトリインスタンスの削除を行わせる。
        std::set<SHGeometryInstance*> delta;
        for (auto it = m_shGeometryInstances.cbegin(); it != m_shGeometryInstances.cend(); ++it)
            delta.insert(*it);
        parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryRemoved, delta);
    }
    
    void TriangleMeshSurfaceNode::setVertices(std::vector<Vertex> &&vertices) {
        m_vertices = vertices;

        optix::Context &optixContext = m_context.getOptiXContext();
        m_optixVertexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, m_vertices.size());
        m_optixVertexBuffer->setElementSize(sizeof(Shared::Vertex));
        {
            auto dstVertices = (Shared::Vertex*)m_optixVertexBuffer->map();
            static_assert(sizeof(Vertex) == sizeof(Shared::Vertex), "These two types must match in size (and structure).");
            std::copy_n((Shared::Vertex*)m_vertices.data(), m_vertices.size(), dstVertices);
            m_optixVertexBuffer->unmap();
        }
    }

    void TriangleMeshSurfaceNode::addMaterialGroup(std::vector<uint32_t> &&indices, const SurfaceMaterialRef &material) {
        optix::Context &optixContext = m_context.getOptiXContext();
        const OptiXProgramSet &progSet = OptiXProgramSets.at(m_context.getID());

        m_sameMaterialGroups.emplace_back(indices);
        std::vector<uint32_t> &sameMaterialGroup = m_sameMaterialGroups.back();
        uint32_t numTriangles = sameMaterialGroup.size() / 3;

        OptiXGeometry geom;
        {
            geom.optixGeometry = optixContext->createGeometry();
            geom.optixGeometry->setPrimitiveCount(numTriangles);
            geom.optixGeometry->setIntersectionProgram(progSet.programIntersectTriangle);
            geom.optixGeometry->setBoundingBoxProgram(progSet.programCalcBBoxForTriangle);

            geom.optixIndexBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, numTriangles);
            geom.optixIndexBuffer->setElementSize(sizeof(Shared::Triangle));
            {
                auto dstTriangles = (Shared::Triangle*)geom.optixIndexBuffer->map();
                for (auto it = sameMaterialGroup.cbegin(); it != sameMaterialGroup.cend(); it += 3)
                    (*dstTriangles++) = Shared::Triangle{ *(it + 0), *(it + 1) , *(it + 2), 0 };
                geom.optixIndexBuffer->unmap();
            }
        }
        m_optixGeometries.push_back(geom);

        m_materials.push_back(material);

        SHGeometryInstance* geomInst = new SHGeometryInstance(m_context);
        {
            optix::GeometryInstance &optixGeomInst = geomInst->getOptiXObject();
            optixGeomInst->setGeometry(geom.optixGeometry);
            optixGeomInst->setMaterialCount(1);
            optixGeomInst->setMaterial(0, material->getOptiXObject());
            optixGeomInst["VLR::pv_vertexBuffer"]->set(m_optixVertexBuffer);
            optixGeomInst["VLR::pv_triangleBuffer"]->set(geom.optixIndexBuffer);
            optixGeomInst["VLR::pv_progDecodeTexCoord"]->set(progSet.callableProgramDecodeTexCoordForTriangle);
            optixGeomInst["VLR::pv_progDecodeHitPoint"]->set(progSet.callableProgramDecodeHitPointForTriangle);
            optixGeomInst["VLR::pv_progFetchAlpha"]->set(m_context.getOptiXCallableProgramNullFetchAlpha());
            optixGeomInst["VLR::pv_progFetchNormal"]->set(m_context.getOptiXCallableProgramNullFetchNormal());
        }
        m_shGeometryInstances.push_back(geomInst);

        // JP: 親にジオメトリインスタンスの追加を行わせる。
        std::set<SHGeometryInstance*> delta;
        delta.insert(geomInst);
        for (auto it = m_parents.cbegin(); it != m_parents.cend(); ++it) {
            ParentNode* parent = *it;
            parent->childUpdateEvent(ParentNode::UpdateEvent::GeometryAdded, delta);
        }
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

    Image2D::Image2D(Context &context, uint32_t width, uint32_t height, DataFormat dataFormat) :
        m_width(width), m_height(height), m_dataFormat(dataFormat) {
        optix::Context &optixContext = context.getOptiXContext();
        switch (m_dataFormat) {
        case VLR::DataFormat::RGB8x3:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE3, width, height);
            break;
        case VLR::DataFormat::RGB_8x4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
            break;
        case VLR::DataFormat::RGBA8x4:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
            break;
        case VLR::DataFormat::Gray8:
            m_optixDataBuffer = optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, width, height);
            break;
        default:
            VLRAssert_ShouldNotBeCalled();
            break;
        }
    }



    LinearImage2D::LinearImage2D(Context &context, const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat) :
    Image2D(context, width, height, Image2D::getInternalFormat(dataFormat)) {
        m_data.resize(getStride() * getWidth() * getHeight());

        switch (dataFormat) {
        case DataFormat::RGB8x3: {
            auto srcHead = (const RGB8x3*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    const RGB8x3 &src = *(srcLineHead + x);
                    RGBA8x4 &dst = *(dstLineHead + x);
                    dst.r = src.r;
                    dst.g = src.g;
                    dst.b = src.b;
                    dst.a = 255;
                }
            }
            break;
        }
        case DataFormat::RGB_8x4: {
            auto srcHead = (const RGB_8x4*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            for (int y = 0; y < height; ++y) {
                auto srcLineHead = srcHead + width * y;
                auto dstLineHead = dstHead + width * y;
                for (int x = 0; x < width; ++x) {
                    const RGB_8x4 &src = *(srcLineHead + x);
                    RGBA8x4 &dst = *(dstLineHead + x);
                    dst.r = src.r;
                    dst.g = src.g;
                    dst.b = src.b;
                    dst.a = 255;
                }
            }
            break;
        }
        case DataFormat::RGBA8x4: {
            auto srcHead = (const RGBA8x4*)linearData;
            auto dstHead = (RGBA8x4*)m_data.data();
            std::copy_n(srcHead, width * height, dstHead);
            break;
        }
        case DataFormat::Gray8: {
            auto srcHead = (const Gray8*)linearData;
            auto dstHead = (Gray8*)m_data.data();
            std::copy_n(srcHead, width * height, dstHead);
            break;
        }
        default:
            VLRAssert(false, "Data format is invalid.");
            break;
        }

        optix::Buffer &buffer = getOptiXObject();
        auto dstData = (uint8_t*)buffer->map();
        {
            std::copy(m_data.cbegin(), m_data.cend(), dstData);
        }
        buffer->unmap();
    }



    FloatTexture::FloatTexture(Context &context) {
        optix::Context &optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }
    
    
    
    Float2Texture::Float2Texture(Context &context) {
        optix::Context &optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }
    
    
    
    Float3Texture::Float3Texture(Context &context) {
        optix::Context &optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }



    Float4Texture::Float4Texture(Context &context) {
        optix::Context &optixContext = context.getOptiXContext();
        m_optixTextureSampler = optixContext->createTextureSampler();
        m_optixTextureSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        m_optixTextureSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR);
        m_optixTextureSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        m_optixTextureSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        m_optixTextureSampler->setMaxAnisotropy(1.0f);
    }
    
    
    
    ImageFloat4Texture::ImageFloat4Texture(Context &context, const Image2DRef &image) :
        Float4Texture(context), m_image(image) {
        m_optixTextureSampler->setBuffer(m_image->getOptiXObject());
    }

    
    
    // static
    void SurfaceMaterial::init(Context &context) {
        MatteSurfaceMaterial::init(context);
        UE4SurfaceMaterial::init(context);
    }

    SurfaceMaterial::SurfaceMaterial(Context &context) {
        optix::Context &optixContext = context.getOptiXContext();
        m_optixMaterial = optixContext->createMaterial();

        m_optixMaterial->setClosestHitProgram(Shared::RayType::Primary, context.getOptiXProgramPathTracingIteration());
        m_optixMaterial->setClosestHitProgram(Shared::RayType::Scattered, context.getOptiXProgramPathTracingIteration());
        m_optixMaterial->setAnyHitProgram(Shared::RayType::Primary, context.getOptiXProgramStochasticAlphaAnyHit());
        m_optixMaterial->setAnyHitProgram(Shared::RayType::Scattered, context.getOptiXProgramStochasticAlphaAnyHit());
        m_optixMaterial->setAnyHitProgram(Shared::RayType::Shadow, context.getOptiXProgramAlphaAnyHit());
    }



    std::map<uint32_t, MatteSurfaceMaterial::OptiXProgramSet> MatteSurfaceMaterial::OptiXProgramSets;
    
    // static
    void MatteSurfaceMaterial::init(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/materials.ptx");

        OptiXProgramSet programSet;

        optix::Context &optixContext = context.getOptiXContext();

        programSet.callableProgramGetBaseColor = optixContext->createProgramFromPTXString(ptx, "VLR::LambertianBRDF_getBaseColor");

        programSet.callableProgramBSDFmatches = optixContext->createProgramFromPTXString(ptx, "VLR::LambertianBRDF_matches");
        programSet.callableProgramSampleBSDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::LambertianBRDF_sampleBSDFInternal");
        programSet.callableProgramEvaluateBSDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::LambertianBRDF_evaluateBSDFInternal");
        programSet.callableProgramEvaluateBSDF_PDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::LambertianBRDF_evaluateBSDF_PDFInternal");

        programSet.callableProgramEvaluateEmittance = optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEmittance");
        programSet.callableProgramEvaluateEDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEDFInternal");

        OptiXProgramSets[context.getID()] = programSet;
    }

    MatteSurfaceMaterial::MatteSurfaceMaterial(Context &context, const Float4TextureRef &texAlbedoRoughness) :
        SurfaceMaterial(context), m_texAlbedoRoughness(texAlbedoRoughness) {
        OptiXProgramSet &progSet = OptiXProgramSets.at(context.getID());

        m_optixMaterial["VLR::pv_progGetBaseColor"]->set(progSet.callableProgramGetBaseColor);

        m_optixMaterial["VLR::pv_progBSDFmatches"]->set(progSet.callableProgramBSDFmatches);
        m_optixMaterial["VLR::pv_progSampleBSDFInternal"]->set(progSet.callableProgramSampleBSDFInternal);
        m_optixMaterial["VLR::pv_progEvaluateBSDFInternal"]->set(progSet.callableProgramEvaluateBSDFInternal);
        m_optixMaterial["VLR::pv_progEvaluateBSDF_PDFInternal"]->set(progSet.callableProgramEvaluateBSDF_PDFInternal);

        m_optixMaterial["VLR::pv_progEvaluateEmittance"]->set(progSet.callableProgramEvaluateEmittance);
        m_optixMaterial["VLR::pv_progEvaluateEDFInternal"]->set(progSet.callableProgramEvaluateEDFInternal);

        m_optixMaterial["VLR::texAlbedoRoughness"]->set(m_texAlbedoRoughness->getOptiXObject());
    }



    std::map<uint32_t, UE4SurfaceMaterial::OptiXProgramSet> UE4SurfaceMaterial::OptiXProgramSets;

    // static
    void UE4SurfaceMaterial::init(Context &context) {
        std::string ptx = readTxtFile("resources/ptxes/materials.ptx");

        OptiXProgramSet programSet;

        optix::Context &optixContext = context.getOptiXContext();

        programSet.callableProgramGetBaseColor = optixContext->createProgramFromPTXString(ptx, "VLR::UE4BRDF_getBaseColor");

        programSet.callableProgramBSDFmatches = optixContext->createProgramFromPTXString(ptx, "VLR::UE4BRDF_matches");
        programSet.callableProgramSampleBSDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::UE4BRDF_sampleBSDFInternal");
        programSet.callableProgramEvaluateBSDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::UE4BRDF_evaluateBSDFInternal");
        programSet.callableProgramEvaluateBSDF_PDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::UE4BRDF_evaluateBSDF_PDFInternal");

        programSet.callableProgramEvaluateEmittance = optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEmittance");
        programSet.callableProgramEvaluateEDFInternal = optixContext->createProgramFromPTXString(ptx, "VLR::NullEDF_evaluateEDFInternal");

        OptiXProgramSets[context.getID()] = programSet;
    }

    UE4SurfaceMaterial::UE4SurfaceMaterial(Context &context, const Float3TextureRef &texBaseColor, const Float2TextureRef &texRoughnessMetallic) :
        SurfaceMaterial(context), m_texBaseColor(texBaseColor), m_texRoughnessMetallic(texRoughnessMetallic) {
        OptiXProgramSet &progSet = OptiXProgramSets.at(context.getID());

        m_optixMaterial["VLR::pv_progGetBaseColor"]->set(progSet.callableProgramGetBaseColor);

        m_optixMaterial["VLR::pv_progBSDFmatches"]->set(progSet.callableProgramBSDFmatches);
        m_optixMaterial["VLR::pv_progSampleBSDFInternal"]->set(progSet.callableProgramSampleBSDFInternal);
        m_optixMaterial["VLR::pv_progEvaluateBSDFInternal"]->set(progSet.callableProgramEvaluateBSDFInternal);
        m_optixMaterial["VLR::pv_progEvaluateBSDF_PDFInternal"]->set(progSet.callableProgramEvaluateBSDF_PDFInternal);

        m_optixMaterial["VLR::pv_progEvaluateEmittance"]->set(progSet.callableProgramEvaluateEmittance);
        m_optixMaterial["VLR::pv_progEvaluateEDFInternal"]->set(progSet.callableProgramEvaluateEDFInternal);

        m_optixMaterial["VLR::texBaseColor"]->set(m_texBaseColor->getOptiXObject());
        m_optixMaterial["VLR::texRoughnessMetallic"]->set(m_texRoughnessMetallic->getOptiXObject());
    }


    
    VLR_API void test() {
        try {
            Context context;

            optix::Context optixContext = context.getOptiXContext();
            optixContext->setRayTypeCount(3);
            optixContext->setEntryPointCount(1);

            SurfaceNode::init(context);
            SurfaceMaterial::init(context);
            
            std::shared_ptr<RootNode> rootNode = createShared<RootNode>(context, createShared<StaticTransform>(translate(0.0f, 0.0f, 0.0f)));

            InternalNodeRef nodeA = createShared<InternalNode>("A", context, createShared<StaticTransform>(translate(-5.0f, 0.0f, 0.0f)));
            InternalNodeRef nodeB = createShared<InternalNode>("B", context, createShared<StaticTransform>(translate(5.0f, 0.0f, 0.0f)));

            rootNode->addChild(nodeA);
            rootNode->addChild(nodeB);

            InternalNodeRef nodeC = createShared<InternalNode>("C", context, createShared<StaticTransform>(translate(0.0f, 0.0f, 0.0f)));

            nodeA->addChild(nodeC);
            nodeB->addChild(nodeC);

            InternalNodeRef nodeD = createShared<InternalNode>("D", context, createShared<StaticTransform>(translate(0.0f, 0.0f, -5.0f)));
            InternalNodeRef nodeE = createShared<InternalNode>("E", context, createShared<StaticTransform>(translate(0.0f, 0.0f, 5.0f)));

            nodeC->addChild(nodeD);
            nodeC->addChild(nodeE);

            std::shared_ptr<TriangleMeshSurfaceNode> meshA = createShared<TriangleMeshSurfaceNode>("MeshA", context);
            {
                std::vector<Vertex> vertices;
                vertices.push_back(Vertex{ Point3D(-1.0f, -1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(0.0f, 1.0f) });
                vertices.push_back(Vertex{ Point3D( 1.0f, -1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(1.0f, 1.0f) });
                vertices.push_back(Vertex{ Point3D( 1.0f,  1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(1.0f, 0.0f) });
                vertices.push_back(Vertex{ Point3D(-1.0f,  1.0f, 0.0f), Normal3D(0, 0, 1), Vector3D(1, 0, 0), TexCoord2D(0.0f, 0.0f) });
                meshA->setVertices(std::move(vertices));

                Image2DRef image;
                {
                    int32_t width, height, n;
                    uint8_t* imageData = stbi_load("resources/R.bmp", &width, &height, &n, 4);
                    image = createShared<LinearImage2D>(context, imageData, width, height, DataFormat::RGBA8x4);
                    stbi_image_free(imageData);
                }
                Float4TextureRef texAlbedoRoughness = createShared<ImageFloat4Texture>(context, image);
                SurfaceMaterialRef matMatte = createShared<MatteSurfaceMaterial>(context, texAlbedoRoughness);

                std::vector<uint32_t> matGroup = {0, 1, 2, 0, 2, 3};
                meshA->addMaterialGroup(std::move(matGroup), matMatte);
            }

            nodeD->addChild(meshA);

            SHGroup &shGroup = rootNode->getSHGroup();

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

            printf("");
        }
        catch (optix::Exception ex) {
            VLRDebugPrintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
        }
    }
}
