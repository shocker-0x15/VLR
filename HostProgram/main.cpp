#include "common.h"

#define NOMINMAX
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "gl_util.h"
// Include glfw3.h after our OpenGL definitions
#include "GLFW/glfw3.h"

#include "scene.h"

#include "../libVLR/utils/cuda_util.h"
#include "StopWatch.h"



#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



static std::filesystem::path getExecutableDirectory() {
    static std::filesystem::path ret;

    static bool done = false;
    if (!done) {
#if defined(HP_Platform_Windows_MSVC)
        TCHAR filepath[1024];
        auto length = GetModuleFileName(NULL, filepath, 1024);
        VLRAssert(length > 0, "Failed to query the executable path.");

        ret = filepath;
#else
        static_assert(false, "Not implemented");
#endif
        ret = ret.remove_filename();

        done = true;
    }

    return ret;
}

static std::string readTxtFile(const std::filesystem::path& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
};



float sRGB_gamma_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
};

float sRGB_degamma_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.04045f)
        return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
};

struct RGB {
    float r, g, b;

    constexpr RGB() : r(0.0f), g(0.0f), b(0.0f) {}
    constexpr RGB(float rr, float gg, float bb) : r(rr), g(gg), b(bb) {}

    RGB operator-() const {
        return RGB(-r, -g, -b);
    }
    RGB operator+(const RGB &v) const {
        return RGB(r + v.r, g + v.g, b + v.b);
    }
    RGB operator-(const RGB &v) const {
        return RGB(r - v.r, g - v.g, b - v.b);
    }
    RGB &operator*=(float s) {
        r *= s;
        g *= s;
        b *= s;
        return *this;
    }

    static constexpr RGB One() { return RGB(1.0f, 1.0f, 1.0f); }
};

struct RGBA {
    RGB rgb;
    float a;

    constexpr RGBA() : a(0.0f) {}
    constexpr RGBA(float rr, float gg, float bb, float aa) : rgb(rr, gg, bb), a(aa) {}
    constexpr RGBA(const RGB &_rgb, float _a) : rgb(_rgb), a(_a) {}
};

RGB exp(const RGB &v) {
    return RGB(std::exp(v.r), std::exp(v.g), std::exp(v.b));
}

RGB sRGB_gamma(const RGB &v) {
    return RGB(sRGB_gamma_s(v.r), sRGB_gamma_s(v.g), sRGB_gamma_s(v.b));
}

RGB min(const RGB &v, float minValue) {
    return RGB(std::fmin(v.r, minValue), std::fmin(v.g, minValue), std::fmin(v.b, minValue));
}

RGB max(const RGB &v, float maxValue) {
    return RGB(std::fmax(v.r, maxValue), std::fmax(v.g, maxValue), std::fmax(v.b, maxValue));
}

static void saveOutputBufferAsImageFile(const vlr::ContextRef &context, const std::string &filenameWoExt, float brightnessCoeff, bool debugRendering) {
    using namespace vlr;

    uint32_t width, height;
    context->getOutputBufferSize(&width, &height);
    auto data = new RGBA[width * height];
    context->readOutputBuffer(reinterpret_cast<float*>(data));

    auto ldrData = new uint32_t[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            RGBA srcPix = data[y * width + x];
            uint32_t &pix = ldrData[y * width + x];

            if (srcPix.rgb.r < -0.001f || srcPix.rgb.g < -0.001f || srcPix.rgb.b < -0.001f)
                hpprintf("Warning: Out of Color Gamut %d, %d: %g, %g, %g\n",
                         x, y, srcPix.rgb.r, srcPix.rgb.g, srcPix.rgb.b);
            srcPix.rgb = max(srcPix.rgb, 0.0f);
            // Simple tone mapping and gamma correction.
            if (!debugRendering) {
                srcPix.rgb *= brightnessCoeff;
                srcPix.rgb = RGB::One() - exp(-srcPix.rgb);
                srcPix.rgb = sRGB_gamma(srcPix.rgb);
            }

            //float Y = srcPix.g;
            //float b = srcPix.r + srcPix.g + srcPix.b;
            //srcPix.r = srcPix.r / b;
            //srcPix.g = srcPix.g / b;
            //srcPix.b = Y * 0.6f;
            //if (srcPix.r > 1.0f || srcPix.g > 1.0f || srcPix.b > 1.0f)
            //    hpprintf("Warning: Over 1.0 %d, %d: %g, %g, %g\n", x, y, srcPix.r, srcPix.g, srcPix.b);
            //srcPix = min(srcPix, 1.0f);

            Assert(srcPix.rgb.r <= 1.0f && srcPix.rgb.g <= 1.0f && srcPix.rgb.b <= 1.0f,
                   "Pixel value should not be greater than 1.0.");

            pix = ((std::min<uint32_t>(srcPix.rgb.r * 256, 255) << 0) |
                   (std::min<uint32_t>(srcPix.rgb.g * 256, 255) << 8) |
                   (std::min<uint32_t>(srcPix.rgb.b * 256, 255) << 16) |
                   (0xFF << 24));
        }
    }

    std::string pngFilename = filenameWoExt + ".png";
    writePNG(pngFilename, width, height, ldrData);

    std::string exrFilename = filenameWoExt + ".exr";
    writeEXR(exrFilename, width, height, reinterpret_cast<const float*>(data));

    delete[] ldrData;
    delete[] data;
}



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



namespace ImGui {
    template <typename EnumType>
    bool RadioButtonE(const char* label, EnumType* v, EnumType v_button) {
        return RadioButton(label, reinterpret_cast<int*>(v), static_cast<int>(v_button));
    }

    bool InputLog2Int(const char* label, int* v, int max_v, int num_digits = 3) {
        float buttonSize = GetFrameHeight();
        float itemInnerSpacingX = GetStyle().ItemInnerSpacing.x;

        BeginGroup();
        PushID(label);

        ImGui::AlignTextToFramePadding();
        SetNextItemWidth(std::max(1.0f, CalcItemWidth() - (buttonSize + itemInnerSpacingX) * 2));
        Text("%s: %*u", label, num_digits, 1 << *v);
        bool changed = false;
        SameLine(0, itemInnerSpacingX);
        if (Button("-", ImVec2(buttonSize, buttonSize))) {
            *v = std::max(*v - 1, 0);
            changed = true;
        }
        SameLine(0, itemInnerSpacingX);
        if (Button("+", ImVec2(buttonSize, buttonSize))) {
            *v = std::min(*v + 1, max_v);
            changed = true;
        }

        PopID();
        EndGroup();

        return changed;
    }
}



class HostProgram {
    static constexpr GLenum s_frameBufferColorFormat = GL_SRGB8_ALPHA8/*GL_RGBA8*/;

    vlr::ContextRef m_context;
    CUstream m_stream[2];
    cudau::Timer m_renderTimer[2];

    GLFWwindow* m_window;
    float m_UIScaling;
    ImGuiStyle m_guiStyleWithGamma;
    ImGuiStyle m_guiStyle;
    
    uint64_t m_frameIndex;
    int32_t m_curFBWidth;
    int32_t m_curFBHeight;

    glu::VertexArray m_vertexArrayForFullScreen;
    glu::Texture2D m_outputBufferGL;
    glu::Sampler m_outputSampler;
    glu::GraphicsProgram m_drawOptiXResultShader;
    glu::FrameBuffer m_frameBuffer;
    glu::GraphicsProgram m_scaleShader;
    glu::Sampler m_scaleSampler;

    bool m_forceLowResolution;

    
    // Trigger Variables
    bool m_resizeRequested;
    int32_t m_requestedSize[2];
    bool m_outputBufferSizeChanged;
    bool m_cameraSettingsChanged;
    bool m_sceneChanged;
    bool m_operatedCameraOnPrevFrame;


    struct KeyState {
        uint64_t timesLastChanged[5];
        bool statesLastChanged[5];
        uint32_t lastIndex;

        KeyState() : lastIndex(0) {
            for (int i = 0; i < 5; ++i) {
                timesLastChanged[i] = 0;
                statesLastChanged[i] = false;
            }
        }

        void recordStateChange(bool state, uint64_t time) {
            bool lastState = statesLastChanged[lastIndex];
            if (state == lastState)
                return;

            lastIndex = (lastIndex + 1) % 5;
            statesLastChanged[lastIndex] = !lastState;
            timesLastChanged[lastIndex] = time;
        }

        bool getState(int32_t goBack = 0) const {
            Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
            return statesLastChanged[(lastIndex + goBack + 5) % 5];
        }

        uint64_t getTime(int32_t goBack = 0) const {
            Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
            return timesLastChanged[(lastIndex + goBack + 5) % 5];
        }
    };

    KeyState m_keyForward;
    KeyState m_keyBackward;
    KeyState m_keyLeftward;
    KeyState m_keyRightward;
    KeyState m_keyUpward;
    KeyState m_keyDownward;
    KeyState m_keyTiltLeft;
    KeyState m_keyTiltRight;
    KeyState m_keyFasterPosMovSpeed;
    KeyState m_keySlowerPosMovSpeed;
    KeyState m_buttonRotate;
    double m_mouseX;
    double m_mouseY;



    uint32_t m_numAccumFrames;

    uint32_t m_renderTargetSizeX;
    uint32_t m_renderTargetSizeY;
    float m_brightnessCoeff;

    vlr::CameraRef m_perspectiveCamera;
    vlr::CameraRef m_equirectangularCamera;
    vlr::CameraRef m_camera;
    int32_t m_cameraTypeIndex;

    vlr::Point3D m_cameraPosition;
    vlr::Quaternion m_cameraOrientation;
    vlr::Quaternion m_tempCameraOrientation;
    float m_cameraPositionalMovingSpeed;
    float m_cameraDirectionalMovingSpeed;
    float m_cameraTiltSpeed;

    float m_persSensitivity;
    float m_fovYInDeg;
    float m_lensRadius;
    float m_focusDistance;

    float m_equiSensitivity;
    float m_phiAngle;
    float m_thetaAngle;

    int32_t m_log2MaxNumAccums = 16;

    float m_environmentRotation;

    bool m_enableDenoiser;
    VLRRenderer m_renderer;
    bool m_enableDebugRendering;
    VLRDebugRenderingMode m_debugRenderingMode;



    Shot m_shot;
    int32_t m_presetViewportIndex;



    void showMiscWindow() {
        ImGui::Begin("Misc", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        //if (m_deviceNames.size() == 1)
        //    ImGui::Text("Device: %s", m_deviceNames[0].c_str());
        //else {
        //    ImGui::Text("Devices:");
        //    for (int i = 0; i < m_deviceNames.size(); ++i)
        //        ImGui::Text("%d: %s", i, m_deviceNames[i].c_str());
        //}

        if (ImGui::InputInt2("Render Size", m_requestedSize, ImGuiInputTextFlags_EnterReturnsTrue))
            m_resizeRequested = true;
        m_outputBufferSizeChanged |= ImGui::Checkbox("Force Low Resolution", &m_forceLowResolution);

        if (ImGui::Button("Save Output")) {
            const char* filename = "output";
            saveOutputBufferAsImageFile(m_context, filename, m_brightnessCoeff, m_enableDebugRendering);
            hpprintf("Image saved: %s\n", filename);
        }

        ImGui::End();
    }

    void showCameraWindow() {
        ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        m_cameraSettingsChanged |= ImGui::InputFloat3("Position", (float*)&m_cameraPosition);
        m_cameraSettingsChanged |= ImGui::InputFloat4("Orientation", (float*)&m_cameraOrientation);

        ImGui::Checkbox("Denoiser", &m_enableDenoiser);

        static constexpr const char* renderers[] = {
            "Path Tracing",
            "Light Tracing",
            "Bidirectional Path Tracing",
        };
        static constexpr const char* debugRenderModes[] = {
            "Base Color",
            "Geometric Normal",
            "Shading Tangent",
            "Shading Bitangent",
            "Shading Normal",
            "Texture Coordinates",
            "Shading Normal View Cos",
            "Geometric vs ShadingNormal",
            "Shading Frame Lengths",
            "Shading Frame Orthogonality",
            "Denoiser Albedo",
            "Denoiser Normal",
        };
        bool rendererChanged = ImGui::Combo("Renderer", (int32_t*)&m_renderer, renderers, lengthof(renderers));
        rendererChanged |= ImGui::Checkbox("Debug Render", &m_enableDebugRendering);
        rendererChanged |= ImGui::Combo("Mode", (int32_t*)&m_debugRenderingMode, debugRenderModes, lengthof(debugRenderModes));
        if (rendererChanged) {
            if (m_enableDebugRendering) {
                ImGui::GetStyle() = m_guiStyle;
                m_context->setRenderer(VLRRenderer_DebugRendering);
                m_context->setDebugRenderingAttribute(m_debugRenderingMode);
            }
            else {
                ImGui::GetStyle() = m_guiStyleWithGamma;
                m_context->setRenderer(m_renderer);
            }
        }
        m_cameraSettingsChanged |= rendererChanged;
        ImGui::SliderFloat("Brightness", &m_brightnessCoeff, 0.01f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

        if (ImGui::InputInt("Viewport", &m_presetViewportIndex)) {
            if (m_presetViewportIndex < 0)
                m_presetViewportIndex = m_shot.viewpoints.size() - 1;
            m_presetViewportIndex %= m_shot.viewpoints.size();

            setViewport(m_shot.viewpoints[m_presetViewportIndex]);

            m_cameraSettingsChanged = true;
        }

        ImGui::Text("Pos. Moving Speed: %g", m_cameraPositionalMovingSpeed);

        static constexpr const char* CameraTypeNames[] = { "Perspective", "Equirectangular" };
        m_cameraSettingsChanged |= ImGui::Combo("Camera Type", &m_cameraTypeIndex, CameraTypeNames, lengthof(CameraTypeNames));

        if (m_cameraTypeIndex == 0) {
            m_cameraSettingsChanged |= ImGui::SliderFloat("fov Y", &m_fovYInDeg, 1, 179, "%.3f", ImGuiSliderFlags_Logarithmic);
            m_cameraSettingsChanged |= ImGui::SliderFloat("Lens Radius", &m_lensRadius, 0.0f, 0.15f, "%.3f", ImGuiSliderFlags_Logarithmic);
            m_cameraSettingsChanged |= ImGui::SliderFloat("Focus Distance", &m_focusDistance, 0.01f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

            m_persSensitivity = m_lensRadius == 0.0f ? 1.0f : 1.0f / (M_PI * m_lensRadius * m_lensRadius);

            m_camera = m_perspectiveCamera;
        }
        else if (m_cameraTypeIndex == 1) {
            m_cameraSettingsChanged |= ImGui::SliderFloat("Phi Angle", &m_phiAngle, M_PI / 18, 2 * M_PI);
            m_cameraSettingsChanged |= ImGui::SliderFloat("Theta Angle", &m_thetaAngle, M_PI / 18, 1 * M_PI);

            m_equiSensitivity = 1.0f / (m_phiAngle * (1 - std::cos(m_thetaAngle)));

            m_camera = m_equirectangularCamera;
        }

        ImGui::InputLog2Int("#MaxNumAccum", &m_log2MaxNumAccums, 16, 5);

        cudau::Timer &renderTimer = m_renderTimer[m_frameIndex % 2];
        float renderTime = NAN;
        if (m_frameIndex >= 2)
            renderTime = renderTimer.report();
        ImGui::Text("%u [spp], %.2f [ms/sample]",
                    std::min(m_numAccumFrames, (1u << m_log2MaxNumAccums)), renderTime);

        ImGui::End();
    }

    void showSceneWindow() {
        using namespace vlr;

        ImGui::Begin("Scene");

        if (ImGui::SliderFloat("Env Rotation", &m_environmentRotation, 0, 360, "%.3f")) {
            m_shot.scene->setEnvironmentRotation(m_environmentRotation * M_PI / 180);
            m_sceneChanged |= true;
        }

        if (ImGui::CollapsingHeader("Scene Outline", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginChild("Hierarchy", ImVec2(-1, 300), false);

            struct SelectedChild {
                InternalNodeRef parent;
                int32_t childIndex;

                bool operator<(const SelectedChild& v) const {
                    if (parent < v.parent) {
                        return true;
                    }
                    else if (parent == v.parent) {
                        if (childIndex < v.childIndex)
                            return true;
                    }
                    return false;
                }
            };

            // JP: 一度にクリックされる要素はひとつだけ。
            // EN: 
            SelectedChild clickedChild{ nullptr, -1 };

            static std::set<SelectedChild> m_selectedNodes;

            const std::function<SelectedChild(InternalNodeRef)> recursiveBuild = [&recursiveBuild](InternalNodeRef parent) {
                SelectedChild clickedChild{ nullptr, -1 };

                std::vector<NodeRef> children;
                uint32_t numChildren = parent->getNumChildren();
                children.resize(numChildren);
                parent->getChildren(numChildren, children.data());

                for (int i = 0; i < numChildren; ++i) {
                    NodeRef child = children[i];
                    SelectedChild curChild{ parent, i };

                    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                    if (m_selectedNodes.count(curChild))
                        node_flags |= ImGuiTreeNodeFlags_Selected;
                    if (child->getType() == "InternalNode") {
                        bool nodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                        bool mouseOnLabel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) > ImGui::GetTreeNodeToLabelSpacing();
                        if (ImGui::IsItemClicked() && mouseOnLabel)
                            clickedChild = curChild;
                        if (nodeOpen) {
                            SelectedChild cSelectedChild = recursiveBuild(std::dynamic_pointer_cast<InternalNodeHolder>(child));
                            if (cSelectedChild.childIndex != -1)
                                clickedChild = cSelectedChild;
                        }
                    }
                    else {
                        node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
                        ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                        if (ImGui::IsItemClicked())
                            clickedChild = curChild;
                    }
                }

                ImGui::TreePop();

                return clickedChild;
            };

            {
                std::vector<NodeRef> rootChildren;
                uint32_t numRootChildren = m_shot.scene->getNumChildren();
                rootChildren.resize(numRootChildren);
                m_shot.scene->getChildren(numRootChildren, rootChildren.data());

                for (int i = 0; i < numRootChildren; ++i) {
                    NodeRef child = rootChildren[i];
                    SelectedChild curChild{ nullptr, i };

                    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                    if (m_selectedNodes.count(curChild))
                        node_flags |= ImGuiTreeNodeFlags_Selected;
                    if (child->getType() == "InternalNode") {
                        bool nodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                        bool mouseOnLabel = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) > ImGui::GetTreeNodeToLabelSpacing();
                        if (ImGui::IsItemClicked() && mouseOnLabel)
                            clickedChild = curChild;
                        if (nodeOpen) {
                            SelectedChild cSelectedChild = recursiveBuild(std::dynamic_pointer_cast<InternalNodeHolder>(child));
                            if (cSelectedChild.childIndex != -1)
                                clickedChild = cSelectedChild;
                        }
                    }
                    else {
                        node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
                        ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, child->getName());
                        if (ImGui::IsItemClicked())
                            clickedChild = curChild;
                    }
                }
            }

            // JP: 何かクリックした要素がある場合。
            // EN: 
            bool newOnlyOneSelected = false;
            if (clickedChild.childIndex != -1) {
                if (ImGui::GetIO().KeyCtrl) {
                    // JP: Ctrlキーを押しながら選択した場合は追加選択or選択解除。
                    // EN: 
                    if (m_selectedNodes.count(clickedChild))
                        m_selectedNodes.erase(clickedChild);
                    else
                        m_selectedNodes.insert(clickedChild);
                }
                else {
                    if (m_selectedNodes.count(clickedChild)) {
                        // JP: クリックした要素を既に選択リストに持っていた場合は全ての選択状態を解除する。
                        //     このとき他に選択要素を持っていた場合はクリックした要素だけを選択状態にする。
                        // EN: 
                        bool multipleSelected = m_selectedNodes.size() > 1;
                        m_selectedNodes.clear();
                        if (multipleSelected)
                            m_selectedNodes.insert(clickedChild);
                    }
                    else {
                        // JP: 全ての選択状態を解除してクリックした要素だけを選択状態にする。
                        // EN: 
                        m_selectedNodes.clear();
                        m_selectedNodes.insert(clickedChild);
                    }
                }

                // JP: クリック時には必ず選択状態に何らかの変化が起きるので、
                //     クリック後に選択要素数が1であれば、必ずそれは新たにひとつだけ選択された要素となる。
                // EN: 
                if (m_selectedNodes.size() == 1)
                    newOnlyOneSelected = true;
            }

            ImGui::EndChild();

            ImGui::Separator();

            NodeRef onlyOneSelectedNode;
            if (m_selectedNodes.size() == 1) {
                const SelectedChild& sc = *m_selectedNodes.cbegin();
                if (sc.parent)
                    onlyOneSelectedNode = sc.parent->getChildAt(sc.childIndex);
                else
                    onlyOneSelectedNode = m_shot.scene->getChildAt(sc.childIndex);
            }

            static char m_nodeName[256];
            static std::string m_nodeType = "";
            static InternalNodeRef m_internalNode;
            static Vector3D m_nodeScale;
            static Vector3D m_nodeRotation;
            static Vector3D m_nodeTranslation;
            if (newOnlyOneSelected) {
                size_t copySize = std::min(std::strlen(onlyOneSelectedNode->getName()), sizeof(m_nodeName) - 1);
                std::memcpy(m_nodeName, onlyOneSelectedNode->getName(), copySize);
                m_nodeName[copySize] = '\0';
                m_nodeType = onlyOneSelectedNode->getType();

                if (m_nodeType == "InternalNode") {
                    m_internalNode = std::dynamic_pointer_cast<InternalNodeHolder>(onlyOneSelectedNode);
                    TransformRef tr = m_internalNode->getTransform();
                    if (tr->getType() == "StaticTransform") {
                        Matrix4x4 mat, invMat;
                        auto sTr = std::dynamic_pointer_cast<StaticTransformHolder>(tr);
                        sTr->getMatrices(&mat, &invMat);

                        mat.decompose(&m_nodeScale, &m_nodeRotation, &m_nodeTranslation);
                        m_nodeRotation *= 180 / M_PI;
                    }
                }

            }
            else if (m_selectedNodes.size() != 1) {
                m_nodeName[0] = '\0';
                m_nodeType = "";
            }

            if (onlyOneSelectedNode) {
                ImGui::AlignTextToFramePadding();
                ImGui::Text("Name:"); ImGui::SameLine();
                ImGui::PushID("NameTextBox");
                if (ImGui::InputText("", m_nodeName, sizeof(m_nodeName), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    onlyOneSelectedNode->setName(m_nodeName);
                }
                ImGui::PopID();

                if (m_selectedNodes.size() == 1) {
                    if (m_nodeType == "InternalNode") {
                        // TODO: tabでフォーカスを動かしたときも編集を確定させる。
                        //       ImGuiにバグがあるっぽい？
                        bool trChanged = false;
                        trChanged |= ImGui::InputFloat3("Scale", (float*)&m_nodeScale, nullptr, ImGuiInputTextFlags_EnterReturnsTrue);
                        trChanged |= ImGui::InputFloat3("Rotation", (float*)&m_nodeRotation, nullptr, ImGuiInputTextFlags_EnterReturnsTrue);
                        trChanged |= ImGui::InputFloat3("Translation", (float*)&m_nodeTranslation, nullptr, ImGuiInputTextFlags_EnterReturnsTrue);
                        if (trChanged) {
                            Matrix4x4 mat = translate<float>(m_nodeTranslation) *
                                rotateZ<float>(m_nodeRotation.z * M_PI / 180) *
                                rotateY<float>(m_nodeRotation.y * M_PI / 180) *
                                rotateX<float>(m_nodeRotation.x * M_PI / 180) *
                                scale<float>(m_nodeScale);

                            auto newTransform = m_context->createStaticTransform(mat);

                            m_internalNode->setTransform(newTransform);

                            m_sceneChanged = true;
                        }
                    }
                }
            }
        }

        ImGui::End();
    }



    void setViewport(const vlr::CameraRef& camera) {
        std::string cameraType = camera->getType();
        if (cameraType == "PerspectiveCamera") {
            camera->get("position", &m_cameraPosition);
            camera->get("orientation", &m_cameraOrientation);
            camera->get("sensitivity", &m_persSensitivity);
            camera->get("fovy", &m_fovYInDeg);
            camera->get("lens radius", &m_lensRadius);
            camera->get("op distance", &m_focusDistance);

            m_perspectiveCamera->set("position", m_cameraPosition);
            m_perspectiveCamera->set("orientation", m_cameraOrientation);
            m_perspectiveCamera->set("aspect", (float)m_renderTargetSizeX / m_renderTargetSizeY);
            m_perspectiveCamera->set("sensitivity", m_persSensitivity);
            m_perspectiveCamera->set("fovy", m_fovYInDeg);
            m_perspectiveCamera->set("lens radius", m_lensRadius);
            m_perspectiveCamera->set("op distance", m_focusDistance);

            m_fovYInDeg *= 180 / M_PI;

            m_camera = m_perspectiveCamera;
            m_cameraTypeIndex = 0;
        }
        else {
            camera->get("position", &m_cameraPosition);
            camera->get("orientation", &m_cameraOrientation);
            camera->get("sensitivity", &m_equiSensitivity);
            camera->get("h angle", &m_phiAngle);
            camera->get("v angle", &m_thetaAngle);

            m_equirectangularCamera->set("position", m_cameraPosition);
            m_equirectangularCamera->set("orientation", m_cameraOrientation);
            m_equirectangularCamera->set("sensitivity", m_equiSensitivity);
            m_equirectangularCamera->set("h angle", m_phiAngle);
            m_equirectangularCamera->set("v angle", m_thetaAngle);

            m_camera = m_equirectangularCamera;
            m_cameraTypeIndex = 1;
        }

        m_tempCameraOrientation = m_cameraOrientation;
    };

public:
    void mouseButtonCallback(int32_t button, int32_t action, int32_t mods) {
        ImGui_ImplGlfw_MouseButtonCallback(m_window, button, action, mods);

        switch (button) {
        case GLFW_MOUSE_BUTTON_MIDDLE: {
            devPrintf("Mouse Middle\n");
            m_buttonRotate.recordStateChange(action == GLFW_PRESS, m_frameIndex);
            break;
        }
        default:
            break;
        }
    }
    void scrollCallback(double x, double y) {

    }
    void cursorPosCallback(double x, double y) {
        m_mouseX = x;
        m_mouseY = y;
    }
    void keyCallback(int32_t key, int32_t scancode, int32_t action, int32_t mods) {
        ImGui_ImplGlfw_KeyCallback(m_window, key, scancode, action, mods);

        switch (key) {
        case GLFW_KEY_W: {
            devPrintf("W: %d\n", action);
            m_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_S: {
            devPrintf("S: %d\n", action);
            m_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_A: {
            devPrintf("A: %d\n", action);
            m_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_D: {
            devPrintf("D: %d\n", action);
            m_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_R: {
            devPrintf("R: %d\n", action);
            m_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_F: {
            devPrintf("F: %d\n", action);
            m_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_Q: {
            devPrintf("Q: %d\n", action);
            m_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_E: {
            devPrintf("E: %d\n", action);
            m_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_T: {
            devPrintf("T: %d\n", action);
            m_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        case GLFW_KEY_G: {
            devPrintf("G: %d\n", action);
            m_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
            break;
        }
        default:
            break;
        }
    }

    void initialize(const vlr::ContextRef& context, GLFWmonitor* monitor, uint32_t initWindowSizeX, uint32_t initWindowSizeY) {
        m_context = context;
        CUcontext cuContext = m_context->getCUcontext();
        CUDADRV_CHECK(cuStreamCreate(&m_stream[0], 0));
        CUDADRV_CHECK(cuStreamCreate(&m_stream[1], 0));

        m_renderTimer[0].initialize(cuContext);
        m_renderTimer[1].initialize(cuContext);

        m_frameIndex = 0;
        m_resizeRequested = false;
        m_operatedCameraOnPrevFrame = false;

        m_forceLowResolution = false;

        m_renderTargetSizeX = initWindowSizeX;
        m_renderTargetSizeY = initWindowSizeY;

        m_enableDenoiser = true;
        m_renderer = VLRRenderer_PathTracing;
        m_enableDebugRendering = false;
        m_debugRenderingMode = VLRDebugRenderingMode_BaseColor;
        if (m_enableDebugRendering)
            m_context->setRenderer(VLRRenderer_DebugRendering);
        else
            m_context->setRenderer(m_renderer);
        m_context->setDebugRenderingAttribute(m_debugRenderingMode);

        constexpr bool enableGLDebugCallback = true;

        // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
        const uint32_t OpenGLMajorVersion = 4;
        const uint32_t OpenGLMinorVersion = 6;
        const char* glsl_version = "#version 460";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
#if defined(Platform_macOS)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        if constexpr (enableGLDebugCallback)
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

        // JP: ウインドウの初期化。
        //     HiDPIディスプレイに対応する。
        float contentScaleX, contentScaleY;
        glfwGetMonitorContentScale(monitor, &contentScaleX, &contentScaleY);
        m_UIScaling = contentScaleX;
        m_window = glfwCreateWindow((int32_t)(m_renderTargetSizeX * m_UIScaling), (int32_t)(m_renderTargetSizeY * m_UIScaling), "VLR", NULL, NULL);
        glfwSetWindowUserPointer(m_window, this);
        if (!m_window) {
            hpprintf("Failed to create a GLFW window.\n");
            glfwTerminate();
            return;
        }

        glfwGetFramebufferSize(m_window, &m_curFBWidth, &m_curFBHeight);

        glfwMakeContextCurrent(m_window);

        glfwSwapInterval(1); // Enable vsync



        // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
        int32_t gl3wRet = gl3wInit();
        if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
            glfwTerminate();
            hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
            return;
        }

        if constexpr (enableGLDebugCallback) {
            glu::enableDebugCallback(true);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, false);
        }



        // Setup ImGui binding
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
        ImGui_ImplGlfw_InitForOpenGL(m_window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Setup style
        ImGui::StyleColorsDark(&m_guiStyle);
        m_guiStyleWithGamma = m_guiStyle;
        const auto degamma = [](const ImVec4 &color) {
            return ImVec4(sRGB_degamma_s(color.x),
                          sRGB_degamma_s(color.y),
                          sRGB_degamma_s(color.z),
                          color.w);
        };
        for (int i = 0; i < ImGuiCol_COUNT; ++i) {
            m_guiStyleWithGamma.Colors[i] = degamma(m_guiStyleWithGamma.Colors[i]);
        }
        ImGui::GetStyle() = m_guiStyleWithGamma;



        m_outputBufferGL.initialize(GL_RGBA32F, m_renderTargetSizeX, m_renderTargetSizeY, 1);
        m_context->bindOutputBuffer(m_renderTargetSizeX, m_renderTargetSizeY, m_outputBufferGL.getHandle());

        m_outputSampler.initialize(glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
                                   glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);

        // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
        m_vertexArrayForFullScreen.initialize();

        const std::filesystem::path exeDir = getExecutableDirectory();

        // JP: OptiXの出力を書き出すシェーダー。
        m_drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "shaders/drawOptiXResult.vert"),
                                               readTxtFile(exeDir / "shaders/drawOptiXResult.frag"));

        // JP: HiDPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
        GLenum colorFormats[] = { s_frameBufferColorFormat };
        GLenum depthFormat = GL_DEPTH_COMPONENT32;
        m_frameBuffer.initialize(m_renderTargetSizeX, m_renderTargetSizeY, 1,
                                 colorFormats, 0, 1,
                                 &depthFormat, false);

        // JP: アップスケール用のシェーダー。
        m_scaleShader.initializeVSPS(readTxtFile(exeDir / "shaders/scale.vert"),
                                     readTxtFile(exeDir / "shaders/scale.frag"));

        // JP: アップスケール用のサンプラー。
        //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
        m_scaleSampler.initialize(glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest, glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);



        glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            auto hostProgram = (HostProgram*)glfwGetWindowUserPointer(window);
            hostProgram->mouseButtonCallback(button, action, mods);
        });
        glfwSetScrollCallback(m_window, [](GLFWwindow* window, double x, double y) {
            auto hostProgram = (HostProgram*)glfwGetWindowUserPointer(window);
            hostProgram->scrollCallback(x, y);
        });
        glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double x, double y) {
            auto hostProgram = (HostProgram*)glfwGetWindowUserPointer(window);
            hostProgram->cursorPosCallback(x, y);
        });
        glfwSetKeyCallback(m_window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            auto hostProgram = (HostProgram*)glfwGetWindowUserPointer(window);
            hostProgram->keyCallback(key, scancode, action, mods);
        });
    }

    void finalize() {
        m_scaleSampler.finalize();
        m_scaleShader.finalize();
        m_frameBuffer.finalize();

        m_drawOptiXResultShader.finalize();
        m_outputBufferGL.finalize();

        m_vertexArrayForFullScreen.finalize();

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(m_window);

        m_renderTimer[1].finalize();
        m_renderTimer[0].finalize();

        CUDADRV_CHECK(cuStreamDestroy(m_stream[1]));
        CUDADRV_CHECK(cuStreamDestroy(m_stream[0]));
    }

    void setShot(const Shot& shot) {
        m_shot = shot;

        m_renderTargetSizeX = shot.renderTargetSizeX;
        m_renderTargetSizeY = shot.renderTargetSizeY;
        m_brightnessCoeff = shot.brightnessCoeff;

        {
            m_perspectiveCamera = m_context->createCamera("Perspective");
            m_persSensitivity = 1.0f;
            m_fovYInDeg = 45;
            m_lensRadius = 0.0f;
            m_focusDistance = 1.0f;
        }
        {
            m_equirectangularCamera = m_context->createCamera("Equirectangular");
            m_equiSensitivity = 1.0f;
            m_phiAngle = 2 * M_PI;
            m_thetaAngle = M_PI;
        }

        setViewport(shot.viewpoints[0]);
        m_presetViewportIndex = 0;
        m_cameraPositionalMovingSpeed = 0.01f;
        m_cameraDirectionalMovingSpeed = 0.0015f;
        m_cameraTiltSpeed = 0.025f;

        m_environmentRotation = shot.environmentRotation * 180 / M_PI;
        m_environmentRotation = m_environmentRotation - std::floor(m_environmentRotation / 360) * 360;

        m_requestedSize[0] = m_renderTargetSizeX;
        m_requestedSize[1] = m_renderTargetSizeY;
        m_resizeRequested = true;
    }

    void run() {
        using namespace vlr;

        StopWatch sw;

        while (!glfwWindowShouldClose(m_window)) {
            CUstream curStream = m_stream[m_frameIndex % 2];
            cudau::Timer &renderTimer = m_renderTimer[m_frameIndex % 2];

            CUDADRV_CHECK(cuStreamSynchronize(curStream));

            glfwPollEvents();

            if (m_resizeRequested) {
                glfwSetWindowSize(m_window,
                                  std::max<int32_t>(360, m_requestedSize[0]) * m_UIScaling,
                                  std::max<int32_t>(360, m_requestedSize[1]) * m_UIScaling);
                m_resizeRequested = false;
            }

            bool resized = false;
            int32_t newFBWidth;
            int32_t newFBHeight;
            glfwGetFramebufferSize(m_window, &newFBWidth, &newFBHeight);
            if (newFBWidth != m_curFBWidth || newFBHeight != m_curFBHeight) {
                m_curFBWidth = newFBWidth;
                m_curFBHeight = newFBHeight;

                m_renderTargetSizeX = m_curFBWidth / m_UIScaling;
                m_renderTargetSizeY = m_curFBHeight / m_UIScaling;
                m_requestedSize[0] = m_renderTargetSizeX;
                m_requestedSize[1] = m_renderTargetSizeY;

                m_frameBuffer.finalize();
                m_outputBufferGL.finalize();

                m_outputBufferGL.initialize(GL_RGBA32F, m_renderTargetSizeX, m_renderTargetSizeY, 1);
                m_context->bindOutputBuffer(m_renderTargetSizeX, m_renderTargetSizeY, m_outputBufferGL.getHandle());

                GLenum colorFormats[] = { s_frameBufferColorFormat };
                GLenum depthFormat = GL_DEPTH_COMPONENT32;
                m_frameBuffer.initialize(m_renderTargetSizeX, m_renderTargetSizeY, 1,
                                         colorFormats, 0, 1,
                                         &depthFormat, false);

                m_perspectiveCamera->set("aspect", (float)m_renderTargetSizeX / m_renderTargetSizeY);

                resized = true;
            }

            bool operatingCamera = false;
            bool cameraIsActuallyMoving = false;

            // process key events
            {
                const auto decideDirection = [](const KeyState& a, const KeyState& b) {
                    int32_t dir = 0;
                    if (a.getState() == true) {
                        if (b.getState() == true)
                            dir = 0;
                        else
                            dir = 1;
                    }
                    else {
                        if (b.getState() == true)
                            dir = -1;
                        else
                            dir = 0;
                    }
                    return dir;
                };

                int32_t trackZ = decideDirection(m_keyForward, m_keyBackward);
                int32_t trackX = decideDirection(m_keyLeftward, m_keyRightward);
                int32_t trackY = decideDirection(m_keyUpward, m_keyDownward);
                int32_t tiltZ = decideDirection(m_keyTiltRight, m_keyTiltLeft);
                int32_t adjustPosMoveSpeed = decideDirection(m_keyFasterPosMovSpeed, m_keySlowerPosMovSpeed);

                m_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
                m_cameraPositionalMovingSpeed = std::min(std::max(m_cameraPositionalMovingSpeed, 1e-6f), 1e+6f);

                static double deltaX = 0, deltaY = 0;
                static double lastX, lastY;
                static double m_prevMouseX = m_mouseX, m_prevMouseY = m_mouseY;
                if (m_buttonRotate.getState() == true) {
                    if (m_buttonRotate.getTime() == m_frameIndex) {
                        lastX = m_mouseX;
                        lastY = m_mouseY;
                    }
                    else {
                        deltaX = m_mouseX - lastX;
                        deltaY = m_mouseY - lastY;
                    }
                }

                float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
                Vector3D axis(deltaY, -deltaX, 0);
                axis /= deltaAngle;
                if (deltaAngle == 0.0f)
                    axis = Vector3D(1, 0, 0);

                m_cameraOrientation = m_cameraOrientation * qRotateZ(m_cameraTiltSpeed * tiltZ);
                m_tempCameraOrientation = m_cameraOrientation * qRotate(m_cameraDirectionalMovingSpeed * deltaAngle, axis);
                m_cameraPosition += m_tempCameraOrientation.toMatrix3x3() * m_cameraPositionalMovingSpeed * Vector3D(trackX, trackY, trackZ);
                if (m_buttonRotate.getState() == false && m_buttonRotate.getTime() == m_frameIndex) {
                    m_cameraOrientation = m_tempCameraOrientation;
                    deltaX = 0;
                    deltaY = 0;
                }

                operatingCamera = (m_keyForward.getState() || m_keyBackward.getState() ||
                    m_keyLeftward.getState() || m_keyRightward.getState() ||
                    m_keyUpward.getState() || m_keyDownward.getState() ||
                    m_keyTiltLeft.getState() || m_keyTiltRight.getState() ||
                    m_buttonRotate.getState());
                cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 || tiltZ != 0 || (m_mouseX != m_prevMouseX) || (m_mouseY != m_prevMouseY)) && operatingCamera;

                m_prevMouseX = m_mouseX;
                m_prevMouseY = m_mouseY;
            }

            {
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                m_outputBufferSizeChanged = resized;
                showMiscWindow();

                m_cameraSettingsChanged = false;
                showCameraWindow();

                m_sceneChanged = false;
                showSceneWindow();

                if (m_cameraTypeIndex == 0) {
                    m_perspectiveCamera->set("position", m_cameraPosition);
                    m_perspectiveCamera->set("orientation", m_tempCameraOrientation);
                    if (m_cameraSettingsChanged) {
                        m_perspectiveCamera->set("aspect", (float)m_renderTargetSizeX / m_renderTargetSizeY);
                        m_perspectiveCamera->set("sensitivity", m_persSensitivity);
                        m_perspectiveCamera->set("fovy", m_fovYInDeg * M_PI / 180);
                        m_perspectiveCamera->set("lens radius", m_lensRadius);
                        m_perspectiveCamera->set("op distance", m_focusDistance);
                    }
                }
                else if (m_cameraTypeIndex == 1) {
                    m_equirectangularCamera->set("position", m_cameraPosition);
                    m_equirectangularCamera->set("orientation", m_tempCameraOrientation);
                    if (m_cameraSettingsChanged) {
                        m_equirectangularCamera->set("sensitivity", m_equiSensitivity);
                        m_equirectangularCamera->set("h angle", m_phiAngle);
                        m_equirectangularCamera->set("v angle", m_thetaAngle);
                    }
                }

                uint32_t shrinkCoeff = (operatingCamera || m_forceLowResolution) ? 4 : 1;

                bool firstFrame =
                    cameraIsActuallyMoving || (m_operatedCameraOnPrevFrame ^ operatingCamera) ||
                    m_outputBufferSizeChanged ||
                    m_cameraSettingsChanged ||
                    m_sceneChanged;
                if (m_frameIndex == 0)
                    firstFrame = true;
                renderTimer.start(curStream);
                uint32_t numAccumFramesLimit = 1u << m_log2MaxNumAccums;
                ImVec2 mousePos = ImGui::GetIO().MousePos;
                m_context->setProbePixel(static_cast<int32_t>(mousePos.x), static_cast<int32_t>(mousePos.y));
                m_context->render(
                    curStream, m_camera, m_enableDenoiser, shrinkCoeff, firstFrame,
                    numAccumFramesLimit, &m_numAccumFrames);
                renderTimer.stop(curStream);

                m_operatedCameraOnPrevFrame = operatingCamera;

                // ----------------------------------------------------------------
                // JP: OptiXの出力とImGuiの描画。

                glBindFramebuffer(GL_FRAMEBUFFER, m_frameBuffer.getHandle(0));
                m_frameBuffer.setDrawBuffers();

                glEnable(GL_FRAMEBUFFER_SRGB);

                glViewport(0, 0, m_frameBuffer.getWidth(), m_frameBuffer.getHeight());
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClearDepth(1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                {
                    glUseProgram(m_drawOptiXResultShader.getHandle());

                    glUniform1i(0, (int32_t)m_renderTargetSizeX);
                    glUniform1f(1, (float)shrinkCoeff);
                    glUniform1f(2, m_brightnessCoeff);
                    glUniform1i(3, (int32_t)m_enableDebugRendering);

                    glBindTextureUnit(0, m_outputBufferGL.getHandle());
                    glBindSampler(0, m_outputSampler.getHandle());

                    glBindVertexArray(m_vertexArrayForFullScreen.getHandle());
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                }

                ImGui::Render();
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                m_frameBuffer.resetDrawBuffers();
                glBindFramebuffer(GL_FRAMEBUFFER, 0);

                // END: draw OptiX's output and ImGui.
                // ----------------------------------------------------------------
            }

            // ----------------------------------------------------------------
            // JP: スケーリング

            if (m_enableDebugRendering)
                glDisable(GL_FRAMEBUFFER_SRGB);
            else
                glEnable(GL_FRAMEBUFFER_SRGB);

            int32_t display_w, display_h;
            glfwGetFramebufferSize(m_window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);

            glUseProgram(m_scaleShader.getHandle());

            glUniform1f(0, m_UIScaling);

            const glu::Texture2D& srcFBTex = m_frameBuffer.getRenderTargetTexture(0, 0);
            glBindTextureUnit(0, srcFBTex.getHandle());
            glBindSampler(0, m_scaleSampler.getHandle());

            glBindVertexArray(m_vertexArrayForFullScreen.getHandle());
            glDrawArrays(GL_TRIANGLES, 0, 3);

            // END: scaling
            // ----------------------------------------------------------------

            glfwSwapBuffers(m_window);

            ++m_frameIndex;
        }
    }
};



static int32_t mainFunc(int32_t argc, const char* argv[]) {
    StopWatch swGlobal;

    swGlobal.start();

    using namespace vlr;

    bool enableLogging = false;
    bool enableGUI = true;
    uint32_t renderImageSizeX = 1920;
    uint32_t renderImageSizeY = 1080;
    uint32_t maxCallableDepth = 8;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--", 2) == 0) {
            if (strcmp(argv[i] + 2, "logging") == 0) {
                enableLogging = true;
            }
            else if (strcmp(argv[i] + 2, "nodisplay") == 0) {
                enableGUI = false;
            }
            else if (strcmp(argv[i] + 2, "imagesize") == 0) {
                ++i;
                renderImageSizeX = atoi(argv[i]);
                ++i;
                renderImageSizeY = atoi(argv[i]);
            }
            else if (strcmp(argv[i] + 2, "maxcallabledepth") == 0) { // TODO: change this to user-friendly parameter.
                ++i;
                if (strncmp(argv[i], "--", 2) != 0)
                    maxCallableDepth = atoi(argv[i]);
            }
        }
    }

    CUcontext cuContext;
    int32_t cuDeviceCount;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

    vlr::ContextRef context = vlr::Context::create(cuContext, enableLogging, maxCallableDepth);

    context->enableAllExceptions();

    Shot shot;
    createScene(context, &shot);

    if (enableGUI) {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) {
            hpprintf("Failed to initialize GLFW.\n");
            return -1;
        }

        GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();

        HostProgram hostProgram;
        hostProgram.initialize(context, primaryMonitor, 640, 640);
        hostProgram.setShot(shot);
        hostProgram.run();
        hostProgram.finalize();

        glfwTerminate();
    }
    else {
        CUstream cuStream;
        CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

        uint32_t renderTargetSizeX = shot.renderTargetSizeX;
        uint32_t renderTargetSizeY = shot.renderTargetSizeY;

        context->bindOutputBuffer(renderTargetSizeX, renderTargetSizeY, 0);

        hpprintf("Setup: %g[s]\n", swGlobal.elapsed(StopWatch::Milliseconds) * 1e-3f);
        swGlobal.start();

        uint32_t numAccumFrames = 0;
        uint32_t imgIndex = 0;
        uint32_t deltaTime = 15 * 1000;
        uint32_t nextTimeToOutput = deltaTime;
        uint32_t finishTime = 123 * 1000 - 3000;
        while (true) {
            context->render(cuStream, shot.viewpoints[0], true,
                            1, numAccumFrames == 0 ? true : false, 0, &numAccumFrames);
            CUDADRV_CHECK(cuStreamSynchronize(cuStream));

            uint64_t elapsed = swGlobal.elapsed(StopWatch::Milliseconds);
            bool finish = swGlobal.elapsedFromRoot(StopWatch::Milliseconds) > finishTime;
            if (elapsed > nextTimeToOutput || finish) {
                char filename[256];
                sprintf(filename, "%03u", imgIndex++);
                saveOutputBufferAsImageFile(context, filename, shot.brightnessCoeff, false);
                hpprintf("%u [spp]: %s, %g [s]\n", numAccumFrames, filename, elapsed * 1e-3f);

                if (finish)
                    break;

                nextTimeToOutput += deltaTime;
                nextTimeToOutput = std::min(nextTimeToOutput, finishTime);
            }
        }

        swGlobal.stop();

        hpprintf("Finish!!: %g[s]\n", swGlobal.stop(StopWatch::Milliseconds) * 1e-3f);
    }

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (const std::exception &ex) {
        hpprintf("Error: %s\n", ex.what());
    }

    return 0;
}
