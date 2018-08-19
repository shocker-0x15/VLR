#pragma once

#include <VLR.h>
#include "basic_types_internal.h"
#include "shared.h"

#include "slot_manager.h"

namespace VLR {
    class Context {
        static uint32_t NextID;
        static uint32_t getInstanceID() {
            return NextID++;
        }

        uint32_t m_ID;
        optix::Context m_optixContext;

        optix::Program m_optixCallableProgramNullFetchAlpha;
        optix::Program m_optixCallableProgramNullFetchNormal;
        optix::Program m_optixCallableProgramFetchAlpha;
        optix::Program m_optixCallableProgramFetchNormal;

        optix::Program m_optixProgramStochasticAlphaAnyHit; // -- Any Hit Program
        optix::Program m_optixProgramAlphaAnyHit; // ------------ Any Hit Program
        optix::Program m_optixProgramPathTracingIteration; // --- Closest Hit Program

        optix::Program m_optixProgramPathTracing; // ------------ Ray Generation Program
        optix::Program m_optixProgramPathTracingMiss; // -------- Miss Program
        optix::Program m_optixProgramException; // -------------- Exception Program

        optix::Buffer m_optixBSDFProcedureSetBuffer;
        uint32_t m_maxNumBSDFProcSet;
        SlotManager m_bsdfProcSetSlotManager;

        optix::Buffer m_optixEDFProcedureSetBuffer;
        uint32_t m_maxNumEDFProcSet;
        SlotManager m_edfProcSetSlotManager;

        optix::Program m_optixCallableProgramNullBSDF_setupBSDF;
        optix::Program m_optixCallableProgramNullBSDF_getBaseColor;
        optix::Program m_optixCallableProgramNullBSDF_matches;
        optix::Program m_optixCallableProgramNullBSDF_sampleBSDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_evaluateBSDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_evaluateBSDF_PDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_weightInternal;
        uint32_t m_nullBSDFProcedureSetIndex;

        optix::Program m_optixCallableProgramNullEDF_setupEDF;
        optix::Program m_optixCallableProgramNullEDF_evaluateEmittanceInternal;
        optix::Program m_optixCallableProgramNullEDF_evaluateEDFInternal;
        uint32_t m_nullEDFProcedureSetIndex;

        optix::Buffer m_optixSurfaceMaterialDescriptorBuffer;
        uint32_t m_maxNumSurfaceMaterialDescriptors;
        SlotManager m_surfMatDescSlotManager;

        optix::Buffer m_outputBuffer;
        optix::Buffer m_rngBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numAccumFrames;

    public:
        Context();
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID);
        void* mapOutputBuffer();
        void unmapOutputBuffer();

        void render(Scene &scene, Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

        const optix::Context &getOptiXContext() const {
            return m_optixContext;
        }

        const optix::Program &getOptiXCallableProgramNullFetchAlpha() const {
            return m_optixCallableProgramNullFetchAlpha;
        }
        const optix::Program &getOptiXCallableProgramNullFetchNormal() const {
            return m_optixCallableProgramNullFetchNormal;
        }
        const optix::Program &getOptiXCallableProgramFetchAlpha() const {
            return m_optixCallableProgramFetchAlpha;
        }
        const optix::Program &getOptiXCallableProgramFetchNormal() const {
            return m_optixCallableProgramFetchNormal;
        }

        // JP: 全マテリアルが共通のClosest Hit, Any Hit Programをバインドする。
        const optix::Program &getOptiXProgramStochasticAlphaAnyHit() const {
            return m_optixProgramStochasticAlphaAnyHit;
        }
        const optix::Program &getOptiXProgramAlphaAnyHit() const {
            return m_optixProgramAlphaAnyHit;
        }
        const optix::Program &getOptiXProgramPathTracingIteration() const {
            return m_optixProgramPathTracingIteration;
        }

        uint32_t setBSDFProcedureSet(const Shared::BSDFProcedureSet &procSet);
        void unsetBSDFProcedureSet(uint32_t index);

        uint32_t setEDFProcedureSet(const Shared::EDFProcedureSet &procSet);
        void unsetEDFProcedureSet(uint32_t index);

        const optix::Program &getOptixCallableProgramNullBSDF_setupBSDF() const {
            return m_optixCallableProgramNullBSDF_setupBSDF;
        }
        uint32_t getNullBSDFProcedureSetIndex() const { return m_nullBSDFProcedureSetIndex; }
        const optix::Program &getOptixCallableProgramNullEDF_setupEDF() const {
            return m_optixCallableProgramNullEDF_setupEDF;
        }
        uint32_t getNullEDFProcedureSetIndex() const { return m_nullEDFProcedureSetIndex; }

        uint32_t setSurfaceMaterialDescriptor(const Shared::SurfaceMaterialDescriptor &matDesc);
        void unsetSurfaceMaterialDescriptor(uint32_t index);
    };



    class ClassIdentifier {
        ClassIdentifier &operator=(const ClassIdentifier &) = delete;

        const ClassIdentifier* m_baseClass;

    public:
        ClassIdentifier(const ClassIdentifier* baseClass) : m_baseClass(baseClass) {}

        const ClassIdentifier* getBaseClass() const {
            return m_baseClass;
        }
    };

    class Object {
    protected:
        Context &m_context;

    public:
        Object(Context &context);
        virtual ~Object() {}

        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier &getClass() const { return ClassID; }

        template <class T>
        bool is() const {
            return &getClass() == &T::ClassID;
        }

        template <class T>
        bool isMemberOf() const {
            const ClassIdentifier* curClass = &getClass();
            while (curClass) {
                if (curClass == &T::ClassID)
                    return true;
                curClass = curClass->getBaseClass();
            }
            return false;
        }

        Context &getContext() {
            return m_context;
        }
    };
}
