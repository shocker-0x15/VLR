#pragma once

#include "common.h"
#include <VLR/VLRCpp.h>



struct EnumTableEntry {
    std::string name;
};



class Parameter {
    VLRCpp::ShaderNodeRef m_shaderNode;
    Parameter* m_parentParameter;
    std::string m_name;

    virtual void drawInternal() const = 0;

public:
    Parameter(const VLRCpp::ShaderNodeRef &shaderNode, Parameter* parentParameter, const std::string& name) : 
        m_shaderNode(shaderNode),
        m_parentParameter(parentParameter),
        m_name(name) {}
    virtual ~Parameter() {}

    virtual void update() const = 0;

    void draw() const {
        drawInternal();
        if (m_parentParameter)
            m_parentParameter->update();
        else
            update();
    }
};



class FloatTupleParameter : public Parameter {
    float* m_values;
    uint32_t m_numValues;

public:
    FloatTupleParameter(const std::string& name, const float* values, uint32_t numValues) :
        Parameter(name),
        m_numValues(numValues) {
        m_values = new float[m_numValues];
        std::copy_n(values, m_numValues, m_values);
    }
    ~FloatTupleParameter() {
        delete[] m_values;
    }

    void draw() const override {
        Assert_NotImplemented();
    }
};



class EnumParameter : public Parameter {
    
    uint32_t m_value;

public:
    EnumParameter(const std::string& name, uint32_t value) :
        Parameter(name),
        m_value(value) {}

    void draw() const override {
        Assert_NotImplemented();
    }
};



class TripletSpectrumParameter : public Parameter {
    EnumParameter m_spectrumType;
    EnumParameter m_colorSpace;
    FloatTupleParameter m_triplet;

public:
    TripletSpectrumParameter(const std::string& name, VLRSpectrumType spectrumType, VLRColorSpace colorSpace, const float triplet[3]) :
        Parameter(name),
        m_spectrumType("spectrum type", spectrumType),
        m_colorSpace("color space", colorSpace),
        m_triplet("triplet", triplet, 3) {
    }
    ~TripletSpectrumParameter() {
    }

    void draw() const override {
        m_spectrumType.draw();
        m_colorSpace.draw();
        m_triplet.draw();
    }
};



class ShaderNode {
    VLRCpp::ShaderNodeRef m_shaderNode;
    Parameter* m_parameters;
    uint32_t m_numParams;

public:
    ShaderNode(const VLRCpp::ShaderNodeRef& shaderNode) : m_shaderNode(shaderNode) {
        m_numParams = m_shaderNode->getNumParameters();
        for (int i = 0; i < m_numParams; ++i) {

        }
    }
    ~ShaderNode() {

    }

    void draw() const {

    }
};



class SurfaceMaterial {
    VLRCpp::SurfaceMaterialRef m_surfaceMaterial;
    Parameter** m_immParams;
    VLRCpp::ShaderNodeSocket* m_nodeParams;
    uint32_t m_numParams;

public:
    SurfaceMaterial(const VLRCpp::SurfaceMaterialRef& surfaceMaterial) : m_surfaceMaterial(surfaceMaterial) {
        m_numParams = m_surfaceMaterial->getNumParameters();

        m_immParams = new Parameter*[m_numParams];
        m_nodeParams = new VLRCpp::ShaderNodeSocket[m_numParams];

        for (int i = 0; i < m_numParams; ++i) {
            Parameter* &immParam = m_immParams[i];
            VLRCpp::ShaderNodeSocket &nodeParam = m_nodeParams[i];

            VLRCpp::ParameterInfo paramInfo = m_surfaceMaterial->getParameterInfo(i);

            const std::string paramName = paramInfo->getName();

            // Node
            // Imm
            // Node | Imm
            VLRSocketForm socketForm = paramInfo->getSocketForm();

            nodeParam = m_surfaceMaterial->getParameterNode(paramName);
            if ((socketForm & VLRSocketForm_ImmediateValue) != 0) {
                // float
                // Spectrum
                VLRParameterType paramType = paramInfo->getType();

                switch (paramType) {
                case VLRParameterType_float: {
                    float value;
                    m_surfaceMaterial->getParameterImmediateValue(paramName, &value);
                    immParam = new FloatTupleParameter();
                    break;
                }
                case VLRParameterType_Spectrum: {
                    VLRTripletSpectrum value;
                    m_surfaceMaterial->getParameterImmediateValue(paramName, &value);
                    immParam = new TripletSpectrumParameter();
                    break;
                }
                default:
                    break;
                }
            }
            else {
                immParam = nullptr;
            }
        }
    }
    ~SurfaceMaterial() {
        delete[] m_nodeParams;
        delete[] m_immParams;
    }

    void draw() const {

    }
};
