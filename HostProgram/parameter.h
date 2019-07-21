#pragma once

#include "common.h"

#define NOMINMAX
#include "imgui.h"

#include <VLR/VLRCpp.h>

class Parameter {
protected:
    VLRCpp::QueryableRef m_parent;
    VLRCpp::ParameterInfo m_info;

public:
    Parameter(const VLRCpp::QueryableRef parent, const VLRCpp::ParameterInfo &info) :
        m_parent(parent), m_info(info) {}
    virtual ~Parameter() {}

    virtual void draw() const = 0;
};

class FloatTupleParameter : public Parameter {
    float* m_values;
    uint32_t m_tupleSize;

public:
    FloatTupleParameter(const VLRCpp::QueryableRef parent, const VLRCpp::ParameterInfo& info) :
        Parameter(parent, info) {
        m_tupleSize = m_info.getTupleSize();
        Assert(m_tupleSize > 0, "Tuple size must be greater than 0.");
        m_values = new float[m_tupleSize];

        parent->get(m_info.getName(), m_values, m_tupleSize);
    }
    ~FloatTupleParameter() {
        delete m_values;
    }

    void draw() const override {
        if (m_tupleSize == 1) {
            ImGui::InputFloat(m_info.getName(), &m_values[0]);
        }
        else {
            const char* param = m_info.getName();
            ImGui::LabelText("%s", param);
            ImGui::PushID(param);
            for (int i = 0; i < m_info.getTupleSize(); ++i) {
                ImGui::InputFloat("", &m_values[i]);
            }
            ImGui::PopID();
        }
    }
};



class ShaderNode {
    VLRCpp::ShaderNodeRef m_shaderNode;
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
