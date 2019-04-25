#include "common.h"

#define NOMINMAX
#include "imgui.h"
#include "imgui_impl_glfw_gl3.h"
#include "GLToolkit.h"
#include "GLFW/glfw3.h"

// DELETE ME
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

// only for catching an exception.
#include <optix_world.h>

#include "scene.h"

#include "StopWatch.h"



uint64_t g_frameIndex;

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
        VLRAssert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        VLRAssert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

KeyState g_keyForward;
KeyState g_keyBackward;
KeyState g_keyLeftward;
KeyState g_keyRightward;
KeyState g_keyUpward;
KeyState g_keyDownward;
KeyState g_keyTiltLeft;
KeyState g_keyTiltRight;
KeyState g_keyFasterPosMovSpeed;
KeyState g_keySlowerPosMovSpeed;
KeyState g_buttonRotate;
double g_mouseX;
double g_mouseY;



bool g_enableDebugRendering;
VLRDebugRenderingMode g_debugRenderingMode;

VLR::Point3D g_cameraPosition;
VLR::Quaternion g_cameraOrientation;
VLR::Quaternion g_tempCameraOrientation;
float g_cameraPositionalMovingSpeed;
float g_cameraDirectionalMovingSpeed;
float g_cameraTiltSpeed;

float g_persSensitivity;
float g_fovYInDeg;
float g_lensRadius;
float g_objPlaneDistance;

float g_equiSensitivity;
float g_phiAngle;
float g_thetaAngle;

uint32_t g_renderTargetSizeX;
uint32_t g_renderTargetSizeY;
float g_brightnessCoeff;
VLRCpp::PerspectiveCameraRef g_perspectiveCamera;
VLRCpp::EquirectangularCameraRef g_equirectangularCamera;
VLRCpp::CameraRef g_camera;
VLRCameraType g_cameraType;

int32_t g_presetViewportIndex;

float g_environmentRotation;



static std::string readTxtFile(const std::string& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
};



float sRGB_gamma_s(float value) {
    VLRAssert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
};

struct RGB {
    float r, g, b;

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

static void saveOutputBufferAsImageFile(const VLRCpp::ContextRef &context, const std::string &filename) {
    using namespace VLR;
    using namespace VLRCpp;

    auto output = (const RGB*)context->mapOutputBuffer();
    uint32_t width, height;
    context->getOutputBufferSize(&width, &height);
    auto data = new uint32_t[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            RGB srcPix = output[y * width + x];
            uint32_t &pix = data[y * width + x];

            if (srcPix.r < 0.0f || srcPix.g < 0.0f || srcPix.b < 0.0f)
                vlrprintf("Warning: Out of Color Gamut %d, %d: %g, %g, %g\n", x, y, srcPix.r, srcPix.g, srcPix.b);
            srcPix *= g_brightnessCoeff;
            srcPix = max(srcPix, 0.0f);
            srcPix = RGB::One() - exp(-srcPix);
            srcPix = sRGB_gamma(srcPix);

            //float Y = srcPix.g;
            //float b = srcPix.r + srcPix.g + srcPix.b;
            //srcPix.r = srcPix.r / b;
            //srcPix.g = srcPix.g / b;
            //srcPix.b = Y * 0.6f;
            //if (srcPix.r > 1.0f || srcPix.g > 1.0f || srcPix.b > 1.0f)
            //    vlrprintf("Warning: Over 1.0 %d, %d: %g, %g, %g\n", x, y, srcPix.r, srcPix.g, srcPix.b);
            //srcPix = min(srcPix, 1.0f);

            Assert(srcPix.r <= 1.0f && srcPix.g <= 1.0f && srcPix.b <= 1.0f, "Pixel value should not be greater than 1.0.");

            pix = ((std::min<uint32_t>(srcPix.r * 256, 255) << 0) |
                   (std::min<uint32_t>(srcPix.g * 256, 255) << 8) |
                   (std::min<uint32_t>(srcPix.b * 256, 255) << 16) |
                   (0xFF << 24));
        }
    }

    stbi_write_bmp(filename.c_str(), width, height, 4, data);
    delete[] data;

    context->unmapOutputBuffer();
}



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



static int32_t mainFunc(int32_t argc, const char* argv[]) {
    StopWatch swGlobal;

    swGlobal.start();

    using namespace VLRCpp;
    using namespace VLR;

    std::set<int32_t> devices;
    bool enableLogging = false;
    bool enableRTX = true;
    bool enableGUI = true;
    uint32_t renderImageSizeX = 1920;
    uint32_t renderImageSizeY = 1080;
    uint32_t maxCallableDepth = 8;
    uint32_t stackSize = 0;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--", 2) == 0) {
            if (strcmp(argv[i] + 2, "list") == 0) {
                vlrPrintDevices();
            }
            else if (strcmp(argv[i] + 2, "devices") == 0) {
                ++i;
                for (; i < argc; ++i) {
                    if (strncmp(argv[i], "--", 2) == 0) {
                        break;
                    }
                    devices.insert(atoi(argv[i]));
                }
                --i;
            }
            else if (strcmp(argv[i] + 2, "disableRTX") == 0) {
                enableRTX = false;
            }
            else if (strcmp(argv[i] + 2, "logging") == 0) {
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
            else if (strcmp(argv[i] + 2, "stacksize") == 0) {
                ++i;
                if (strncmp(argv[i], "--", 2) != 0)
                    stackSize = atoi(argv[i]);
            }
        }
    }

    int32_t primaryDevice = 0;
    std::vector<int32_t> deviceArray;
    if (!devices.empty()) {
        for (auto it = devices.cbegin(); it != devices.cend(); ++it)
            deviceArray.push_back(*it);

        primaryDevice = deviceArray.front();
    }

    char deviceName[128];
    vlrGetDeviceName(primaryDevice, deviceName, lengthof(deviceName));

    VLRCpp::ContextRef context = VLRCpp::Context::create(enableLogging, enableRTX, maxCallableDepth, stackSize,
                                                         deviceArray.empty() ? nullptr : deviceArray.data(), deviceArray.size());

    Shot shot;
    createScene(context, &shot);

    g_renderTargetSizeX = shot.renderTargetSizeX;
    g_renderTargetSizeY = shot.renderTargetSizeY;
    g_brightnessCoeff = shot.brightnessCoeff;

    {
        g_perspectiveCamera = context->createPerspectiveCamera();
        g_persSensitivity = 1.0f;
        g_fovYInDeg = 45;
        g_lensRadius = 0.0f;
        g_objPlaneDistance = 1.0f;
    }
    {
        g_equirectangularCamera = context->createEquirectangularCamera();
        g_equiSensitivity = 1.0f;
        g_phiAngle = 2 * M_PI;
        g_thetaAngle = M_PI;
    }

    const auto setViewport = [&](const VLRCpp::CameraRef &camera) {
        g_cameraType = camera->getCameraType();
        if (g_cameraType == VLRCameraType_Perspective) {
            auto viewport = std::dynamic_pointer_cast<VLRCpp::PerspectiveCameraHolder>(camera);

            viewport->getPosition(&g_cameraPosition);
            viewport->getOrientation(&g_cameraOrientation);
            viewport->getSensitivity(&g_persSensitivity);
            viewport->getFovY(&g_fovYInDeg);
            viewport->getLensRadius(&g_lensRadius);
            viewport->getObjectPlaneDistance(&g_objPlaneDistance);

            g_perspectiveCamera->setPosition(g_cameraPosition);
            g_perspectiveCamera->setOrientation(g_cameraOrientation);
            g_perspectiveCamera->setAspectRatio((float)g_renderTargetSizeX / g_renderTargetSizeY);
            g_perspectiveCamera->setSensitivity(g_persSensitivity);
            g_perspectiveCamera->setFovY(g_fovYInDeg);
            g_perspectiveCamera->setLensRadius(g_lensRadius);
            g_perspectiveCamera->setObjectPlaneDistance(g_objPlaneDistance);

            g_fovYInDeg *= 180 / M_PI;

            g_camera = g_perspectiveCamera;
        }
        else {
            auto viewport = std::dynamic_pointer_cast<VLRCpp::EquirectangularCameraHolder>(camera);

            viewport->getPosition(&g_cameraPosition);
            viewport->getOrientation(&g_cameraOrientation);
            viewport->getSensitivity(&g_equiSensitivity);
            viewport->getAngles(&g_phiAngle, &g_thetaAngle);

            g_equirectangularCamera->setPosition(g_cameraPosition);
            g_equirectangularCamera->setOrientation(g_cameraOrientation);
            g_equirectangularCamera->setSensitivity(g_equiSensitivity);
            g_equirectangularCamera->setAngles(g_phiAngle, g_thetaAngle);

            g_camera = g_equirectangularCamera;
        }

        g_tempCameraOrientation = g_cameraOrientation;
    };

    setViewport(shot.viewpoints[0]);
    g_presetViewportIndex = 0;
    g_cameraPositionalMovingSpeed = 0.01f;
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;

    g_enableDebugRendering = false;
    g_debugRenderingMode = VLRDebugRenderingMode_GeometricNormal;

    g_environmentRotation = shot.environmentRotation * 180 / M_PI;
    g_environmentRotation = g_environmentRotation - std::floor(g_environmentRotation / 360) * 360;



    if (enableGUI) {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return 1;

        GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();

        // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
        const uint32_t OpenGLMajorVersion = 4;
        const uint32_t OpenGLMinorVersion = 6;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(Platform_macOS)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        // JP: ウインドウの初期化。
        //     HiDPIディスプレイに対応する。
        float contentScaleX, contentScaleY;
        glfwGetMonitorContentScale(primaryMonitor, &contentScaleX, &contentScaleY);
        const float UIScaling = contentScaleX;
        GLFWwindow* window = glfwCreateWindow((int32_t)(g_renderTargetSizeX * UIScaling), (int32_t)(g_renderTargetSizeY * UIScaling), "VLR", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return -1;
        }

        int32_t curFBWidth;
        int32_t curFBHeight;
        glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

        glfwMakeContextCurrent(window);

        glfwSwapInterval(1); // Enable vsync

        // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
        int32_t gl3wRet = gl3wInit();
        if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
            glfwTerminate();
            hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
            return -1;
        }



        // Setup ImGui binding
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
        ImGui_ImplGlfwGL3_Init(window, true);

        // Setup style
        ImGui::StyleColorsDark();



        // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
        GLTK::VertexArray vertexArrayForFullScreen;
        vertexArrayForFullScreen.initialize();

        GLTK::Buffer outputBufferGL;
        outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(RGB), g_renderTargetSizeX * g_renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);

        context->bindOutputBuffer(g_renderTargetSizeX, g_renderTargetSizeY, outputBufferGL.getRawHandle());

        GLTK::BufferTexture outputTexture;
        outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGB32F);

        // JP: OptiXの出力を書き出すシェーダー。
        GLTK::GraphicsShader drawOptiXResultShader;
        drawOptiXResultShader.initializeVSPS(readTxtFile("resources/shaders/drawOptiXResult.vert"),
                                             readTxtFile("resources/shaders/drawOptiXResult.frag"));

        // JP: HiDPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
        GLTK::FrameBuffer frameBuffer;
        frameBuffer.initialize(g_renderTargetSizeX, g_renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

        // JP: アップスケール用のシェーダー。
        GLTK::GraphicsShader scaleShader;
        scaleShader.initializeVSPS(readTxtFile("resources/shaders/scale.vert"),
                                   readTxtFile("resources/shaders/scale.frag"));

        // JP: アップスケール用のサンプラー。
        //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
        GLTK::Sampler scaleSampler;
        scaleSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest, GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



        glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

            switch (button) {
            case GLFW_MOUSE_BUTTON_MIDDLE: {
                devPrintf("Mouse Middle\n");
                g_buttonRotate.recordStateChange(action == GLFW_PRESS, g_frameIndex);
                break;
            }
            default:
                break;
            }
        });
        glfwSetScrollCallback(window, [](GLFWwindow* window, double x, double y) {
            //devPrintf("%g, %g\n", x, y);
        });
        glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
            g_mouseX = x;
            g_mouseY = y;
        });
        glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

            switch (key) {
            case GLFW_KEY_W: {
                devPrintf("W: %d\n", action);
                g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_S: {
                devPrintf("S: %d\n", action);
                g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_A: {
                devPrintf("A: %d\n", action);
                g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_D: {
                devPrintf("D: %d\n", action);
                g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_R: {
                devPrintf("R: %d\n", action);
                g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_F: {
                devPrintf("F: %d\n", action);
                g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_Q: {
                devPrintf("Q: %d\n", action);
                g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_E: {
                devPrintf("E: %d\n", action);
                g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_T: {
                devPrintf("T: %d\n", action);
                g_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            case GLFW_KEY_G: {
                devPrintf("G: %d\n", action);
                g_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
                break;
            }
            default:
                break;
            }
        });

        StopWatch sw;
        uint64_t accumFrameTimes = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            bool operatingCamera = false;
            bool cameraIsActuallyMoving = false;

            static bool g_resizeRequested = false;
            static int32_t g_requestedSize[2] = { g_renderTargetSizeX, g_renderTargetSizeY };
            if (g_resizeRequested) {
                glfwSetWindowSize(window, 
                                  std::max<int32_t>(360, g_requestedSize[0]) * UIScaling, 
                                  std::max<int32_t>(360, g_requestedSize[1]) * UIScaling);
                g_resizeRequested = false;
            }
            bool resized = false;
            int32_t newFBWidth;
            int32_t newFBHeight;
            glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
            if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
                curFBWidth = newFBWidth;
                curFBHeight = newFBHeight;

                g_renderTargetSizeX = curFBWidth / UIScaling;
                g_renderTargetSizeY = curFBHeight / UIScaling;
                g_requestedSize[0] = g_renderTargetSizeX;
                g_requestedSize[1] = g_renderTargetSizeY;

                frameBuffer.finalize();
                outputTexture.finalize();
                outputBufferGL.finalize();

                outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(RGB), g_renderTargetSizeX * g_renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);

                context->bindOutputBuffer(g_renderTargetSizeX, g_renderTargetSizeY, outputBufferGL.getRawHandle());

                outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGB32F);

                frameBuffer.initialize(g_renderTargetSizeX, g_renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

                g_perspectiveCamera->setAspectRatio((float)g_renderTargetSizeX / g_renderTargetSizeY);

                resized = true;
            }

            // process key events
            {
                const auto decideDirection = [](const KeyState &a, const KeyState &b) {
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

                int32_t trackZ = decideDirection(g_keyForward, g_keyBackward);
                int32_t trackX = decideDirection(g_keyLeftward, g_keyRightward);
                int32_t trackY = decideDirection(g_keyUpward, g_keyDownward);
                int32_t tiltZ = decideDirection(g_keyTiltRight, g_keyTiltLeft);
                int32_t adjustPosMoveSpeed = decideDirection(g_keyFasterPosMovSpeed, g_keySlowerPosMovSpeed);

                g_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
                g_cameraPositionalMovingSpeed = std::min(std::max(g_cameraPositionalMovingSpeed, 1e-6f), 1e+6f);

                static double deltaX = 0, deltaY = 0;
                static double lastX, lastY;
                static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
                if (g_buttonRotate.getState() == true) {
                    if (g_buttonRotate.getTime() == g_frameIndex) {
                        lastX = g_mouseX;
                        lastY = g_mouseY;
                    }
                    else {
                        deltaX = g_mouseX - lastX;
                        deltaY = g_mouseY - lastY;
                    }
                }

                float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
                Vector3D axis(deltaY, -deltaX, 0);
                axis /= deltaAngle;
                if (deltaAngle == 0.0f)
                    axis = Vector3D(1, 0, 0);

                g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
                g_tempCameraOrientation = g_cameraOrientation * qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
                g_cameraPosition += g_tempCameraOrientation.toMatrix3x3() * g_cameraPositionalMovingSpeed * Vector3D(trackX, trackY, trackZ);
                if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == g_frameIndex) {
                    g_cameraOrientation = g_tempCameraOrientation;
                    deltaX = 0;
                    deltaY = 0;
                }

                operatingCamera = (g_keyForward.getState() || g_keyBackward.getState() ||
                                   g_keyLeftward.getState() || g_keyRightward.getState() ||
                                   g_keyUpward.getState() || g_keyDownward.getState() ||
                                   g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                                   g_buttonRotate.getState());
                cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 || tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY)) && operatingCamera;

                g_prevMouseX = g_mouseX;
                g_prevMouseY = g_mouseY;
            }

            {
                ImGui_ImplGlfwGL3_NewFrame(g_renderTargetSizeX, g_renderTargetSizeY, UIScaling);

                bool outputBufferSizeChanged = resized;
                static bool g_forceLowResolution = false;
                {
                    ImGui::Begin("Misc", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                    ImGui::Text("Device: %s", deviceName);

                    if (ImGui::InputInt2("Render Size", g_requestedSize, ImGuiInputTextFlags_EnterReturnsTrue))
                        g_resizeRequested = true;
                    outputBufferSizeChanged |= ImGui::Checkbox("Force Low Resolution", &g_forceLowResolution);

                    if (ImGui::Button("Save Output"))
                        saveOutputBufferAsImageFile(context, "output.bmp");

                    ImGui::End();
                }

                bool cameraSettingsChanged = false;
                static uint32_t g_numAccumFrames = 1;
                {
                    ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                    cameraSettingsChanged |= ImGui::InputFloat3("Position", (float*)&g_cameraPosition);
                    cameraSettingsChanged |= ImGui::InputFloat4("Orientation", (float*)&g_cameraOrientation);

                    const char* debugRenderModes[] = {
                        "GeometricNormal",
                        "ShadingTangent",
                        "ShadingBitangent",
                        "ShadingNormal",
                        "TC0Direction",
                        "TextureCoordinates",
                        "GeometricVsShadingNormal",
                        "ShadingFrameLengths",
                        "ShadingFrameOrthogonality",
                    };
                    cameraSettingsChanged |= ImGui::Checkbox("Debug Render", &g_enableDebugRendering);
                    cameraSettingsChanged |= ImGui::Combo("Mode", (int32_t*)&g_debugRenderingMode, debugRenderModes, lengthof(debugRenderModes));
                    ImGui::SliderFloat("Brightness", &g_brightnessCoeff, 0.01f, 10.0f, "%.3f", 2.0f);

                    if (ImGui::InputInt("Viewport", &g_presetViewportIndex)) {
                        if (g_presetViewportIndex < 0)
                            g_presetViewportIndex = shot.viewpoints.size() - 1;
                        g_presetViewportIndex %= shot.viewpoints.size();

                        setViewport(shot.viewpoints[g_presetViewportIndex]);

                        cameraSettingsChanged = true;
                    }

                    ImGui::Text("Pos. Moving Speed: %g", g_cameraPositionalMovingSpeed);

                    const char* CameraTypeNames[] = { "Perspective", "Equirectangular" };
                    cameraSettingsChanged |= ImGui::Combo("Camera Type", (int32_t*)&g_cameraType, CameraTypeNames, lengthof(CameraTypeNames));

                    if (g_cameraType == VLRCameraType_Perspective) {
                        cameraSettingsChanged |= ImGui::SliderFloat("fov Y", &g_fovYInDeg, 1, 179, "%.3f", 2.0f);
                        cameraSettingsChanged |= ImGui::SliderFloat("Lens Radius", &g_lensRadius, 0.0f, 0.15f, "%.3f", 1.0f);
                        cameraSettingsChanged |= ImGui::SliderFloat("Object Plane Distance", &g_objPlaneDistance, 0.01f, 20.0f, "%.3f", 2.0f);

                        g_persSensitivity = g_lensRadius == 0.0f ? 1.0f : 1.0f / (M_PI * g_lensRadius * g_lensRadius);

                        g_camera = g_perspectiveCamera;
                    }
                    else if (g_cameraType == VLRCameraType_Equirectangular) {
                        cameraSettingsChanged |= ImGui::SliderFloat("Phi Angle", &g_phiAngle, M_PI / 18, 2 * M_PI);
                        cameraSettingsChanged |= ImGui::SliderFloat("Theta Angle", &g_thetaAngle, M_PI / 18, 1 * M_PI);

                        g_equiSensitivity = 1.0f / (g_phiAngle * (1 - std::cos(g_thetaAngle)));

                        g_camera = g_equirectangularCamera;
                    }

                    ImGui::Text("%u [spp], %g [ms/sample]", g_numAccumFrames, (float)accumFrameTimes / (g_numAccumFrames - 1));

                    ImGui::End();
                }

                bool sceneChanged = false;
                {
                    ImGui::Begin("Scene");

                    if (ImGui::SliderFloat("Env Rotation", &g_environmentRotation, 0, 360, "%.3f")) {
                        shot.scene->setEnvironmentRotation(g_environmentRotation * M_PI / 180);
                        sceneChanged |= true;
                    }

                    if (ImGui::CollapsingHeader("Scene Outline", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::BeginChild("Hierarchy", ImVec2(-1, 300), false);

                        struct SelectedChild {
                            InternalNodeRef parent;
                            int32_t childIndex;

                            bool operator<(const SelectedChild &v) const {
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

                        static std::set<SelectedChild> g_selectedNodes;

                        const std::function<SelectedChild(InternalNodeRef)> recursiveBuild = [&recursiveBuild](InternalNodeRef parent) {
                            SelectedChild clickedChild{ nullptr, -1 };

                            for (int i = 0; i < parent->getNumChildren(); ++i) {
                                NodeRef child = parent->getChildAt(i);
                                SelectedChild curChild{ parent, i };

                                ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                                if (g_selectedNodes.count(curChild))
                                    node_flags |= ImGuiTreeNodeFlags_Selected;
                                if (child->getNodeType() == VLRNodeType_InternalNode) {
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

                        SelectedChild clickedChild{ nullptr, -1 };

                        for (int i = 0; i < shot.scene->getNumChildren(); ++i) {
                            NodeRef child = shot.scene->getChildAt(i);
                            SelectedChild curChild{ nullptr, i };

                            ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                            if (g_selectedNodes.count(curChild))
                                node_flags |= ImGuiTreeNodeFlags_Selected;
                            if (child->getNodeType() == VLRNodeType_InternalNode) {
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

                        // JP: 何かクリックした要素がある場合。
                        bool newOnlyOneSelected = false;
                        if (clickedChild.childIndex != -1) {
                            if (ImGui::GetIO().KeyCtrl) {
                                // JP: Ctrlキーを押しながら選択した場合は追加選択or選択解除。
                                if (g_selectedNodes.count(clickedChild))
                                    g_selectedNodes.erase(clickedChild);
                                else
                                    g_selectedNodes.insert(clickedChild);
                            }
                            else {
                                if (g_selectedNodes.count(clickedChild)) {
                                    // JP: クリックした要素を既に選択リストに持っていた場合は全ての選択状態を解除する。
                                    //     このとき他に選択要素を持っていた場合はクリックした要素だけを選択状態にする。
                                    bool multiplySelected = g_selectedNodes.size() > 1;
                                    g_selectedNodes.clear();
                                    if (multiplySelected)
                                        g_selectedNodes.insert(clickedChild);
                                }
                                else {
                                    // JP: 全ての選択状態を解除してクリックした要素だけを選択状態にする。
                                    g_selectedNodes.clear();
                                    g_selectedNodes.insert(clickedChild);
                                }
                            }

                            // JP: クリック時には必ず選択状態に何らかの変化が起きるので、
                            //     クリック後に選択要素数が1であれば、必ずそれは新たにひとつだけ選択された要素となる。
                            if (g_selectedNodes.size() == 1)
                                newOnlyOneSelected = true;
                        }

                        ImGui::EndChild();

                        ImGui::Separator();

                        NodeRef node;

                        if (g_selectedNodes.size() == 1) {
                            const SelectedChild &sc = *g_selectedNodes.cbegin();
                            if (sc.parent)
                                node = sc.parent->getChildAt(sc.childIndex);
                            else
                                node = shot.scene->getChildAt(sc.childIndex);
                        }

                        static char g_nodeName[256];
                        if (newOnlyOneSelected) {
                            size_t copySize = std::min(std::strlen(node->getName()), sizeof(g_nodeName) - 1);
                            std::memcpy(g_nodeName, node->getName(), copySize);
                            g_nodeName[copySize] = '\0';
                        }
                        else if (g_selectedNodes.size() != 1) {
                            g_nodeName[0] = '\0';
                        }

                        if (node) {
                            ImGui::AlignTextToFramePadding();
                            ImGui::Text("Name:"); ImGui::SameLine();
                            ImGui::PushID("NameTextBox");
                            if (ImGui::InputText("", g_nodeName, sizeof(g_nodeName), ImGuiInputTextFlags_EnterReturnsTrue)) {
                                node->setName(g_nodeName);
                            }
                            ImGui::PopID();

                            if (node->getNodeType() == VLRNodeType_InternalNode) {

                            }
                            else {

                            }
                        }
                    }

                    ImGui::End();
                }

                if (g_cameraType == VLRCameraType_Perspective) {
                    g_perspectiveCamera->setPosition(g_cameraPosition);
                    g_perspectiveCamera->setOrientation(g_tempCameraOrientation);
                    if (cameraSettingsChanged) {
                        g_perspectiveCamera->setAspectRatio((float)g_renderTargetSizeX / g_renderTargetSizeY);
                        g_perspectiveCamera->setSensitivity(g_persSensitivity);
                        g_perspectiveCamera->setFovY(g_fovYInDeg * M_PI / 180);
                        g_perspectiveCamera->setLensRadius(g_lensRadius);
                        g_perspectiveCamera->setObjectPlaneDistance(g_objPlaneDistance);
                    }
                }
                else if (g_cameraType == VLRCameraType_Equirectangular) {
                    g_equirectangularCamera->setPosition(g_cameraPosition);
                    g_equirectangularCamera->setOrientation(g_tempCameraOrientation);
                    if (cameraSettingsChanged) {
                        g_equirectangularCamera->setSensitivity(g_equiSensitivity);
                        g_equirectangularCamera->setAngles(g_phiAngle, g_thetaAngle);
                    }
                }

                static bool g_operatedCameraOnPrevFrame = false;
                uint32_t shrinkCoeff = (operatingCamera || g_forceLowResolution) ? 4 : 1;

                bool firstFrame = cameraIsActuallyMoving || (g_operatedCameraOnPrevFrame ^ operatingCamera) || outputBufferSizeChanged || cameraSettingsChanged || sceneChanged;
                if (g_frameIndex == 0)
                    firstFrame = true;
                if (firstFrame)
                    accumFrameTimes = 0;
                else
                    sw.start();
                if (g_enableDebugRendering)
                    context->debugRender(shot.scene, g_camera, g_debugRenderingMode, shrinkCoeff, firstFrame, &g_numAccumFrames);
                else
                    context->render(shot.scene, g_camera, shrinkCoeff, firstFrame, &g_numAccumFrames);
                if (!firstFrame)
                    accumFrameTimes += sw.stop(StopWatch::Milliseconds);

                //// DELETE ME
                //if (g_numAccumFrames == 32) {
                //    devPrintf("Camera:\n");
                //    devPrintf("Position: %g, %g, %g\n", g_cameraPos.x, g_cameraPos.y, g_cameraPos.z);
                //    devPrintf("Orientation: %g, %g, %g, %g\n", g_cameraOrientation.x, g_cameraOrientation.y, g_cameraOrientation.z, g_cameraOrientation.w);

                //    auto output = (const RGBSpectrum*)context->mapOutputBuffer();
                //    auto data = new uint32_t[renderTargetSizeX * renderTargetSizeY];
                //    for (int y = 0; y < renderTargetSizeY; ++y) {
                //        for (int x = 0; x < renderTargetSizeX; ++x) {
                //            RGBSpectrum srcPix = output[y * renderTargetSizeX + x];
                //            uint32_t &pix = data[y * renderTargetSizeX + x];

                //            srcPix *= g_brightnessCoeff;
                //            srcPix = RGBSpectrum::One() - exp(-srcPix);
                //            srcPix = sRGB_gamma(srcPix);

                //            pix = ((std::min<uint8_t>(srcPix.r * 256, 255) << 0) |
                //                   (std::min<uint8_t>(srcPix.g * 256, 255) << 8) |
                //                   (std::min<uint8_t>(srcPix.b * 256, 255) << 16) |
                //                   (0xFF << 24));
                //        }
                //    }
                //    stbi_write_png("output.png", renderTargetSizeX, renderTargetSizeY, 4, data, sizeof(data[0]) * renderTargetSizeX);
                //    delete[] data;
                //    context->unmapOutputBuffer();
                //}

                g_operatedCameraOnPrevFrame = operatingCamera;

                // ----------------------------------------------------------------
                // JP: OptiXの出力とImGuiの描画。

                frameBuffer.bind(GLTK::FrameBuffer::Target::ReadDraw);

                glViewport(0, 0, frameBuffer.getWidth(), frameBuffer.getHeight());
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClearDepth(1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                {
                    drawOptiXResultShader.useProgram();

                    glUniform1i(0, (int32_t)g_renderTargetSizeX); GLTK::errorCheck();

                    glUniform1f(1, (float)shrinkCoeff); GLTK::errorCheck();

                    glUniform1f(2, g_brightnessCoeff); GLTK::errorCheck();

                    glActiveTexture(GL_TEXTURE0); GLTK::errorCheck();
                    outputTexture.bind();

                    vertexArrayForFullScreen.bind();
                    glDrawArrays(GL_TRIANGLES, 0, 3); GLTK::errorCheck();
                    vertexArrayForFullScreen.unbind();

                    outputTexture.unbind();
                }

                ImGui::Render();
                ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());

                frameBuffer.unbind();

                // END: draw OptiX's output and ImGui.
                // ----------------------------------------------------------------
            }

            // ----------------------------------------------------------------
            // JP: スケーリング

            int32_t display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);

            scaleShader.useProgram();

            glUniform1f(0, UIScaling);

            glActiveTexture(GL_TEXTURE0);
            GLTK::Texture2D &srcFBTex = frameBuffer.getRenderTargetTexture();
            srcFBTex.bind();
            scaleSampler.bindToTextureUnit(0);

            vertexArrayForFullScreen.bind();
            glDrawArrays(GL_TRIANGLES, 0, 3);
            vertexArrayForFullScreen.unbind();

            srcFBTex.unbind();

            // END: scaling
            // ----------------------------------------------------------------

            glfwSwapBuffers(window);

            ++g_frameIndex;
        }

        scaleSampler.finalize();
        scaleShader.finalize();
        frameBuffer.finalize();

        drawOptiXResultShader.finalize();
        outputTexture.finalize();
        outputBufferGL.finalize();

        vertexArrayForFullScreen.finalize();

        ImGui_ImplGlfwGL3_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    else {
        uint32_t renderTargetSizeX = renderImageSizeX;
        uint32_t renderTargetSizeY = renderImageSizeY;

        context->bindOutputBuffer(renderTargetSizeX, renderTargetSizeY, 0);

        vlrprintf("Setup: %g[s]\n", swGlobal.elapsed(StopWatch::Milliseconds) * 1e-3f);
        swGlobal.start();

        uint32_t numAccumFrames = 0;
        uint32_t imgIndex = 0;
        uint32_t deltaTime = 15 * 1000;
        uint32_t nextTimeToOutput = deltaTime;
        uint32_t finishTime = 123 * 1000 - 3000;
        auto data = new uint32_t[renderTargetSizeX * renderTargetSizeY];
        while (true) {
            context->render(shot.scene, g_camera, 1, numAccumFrames == 0 ? true : false, &numAccumFrames);

            uint64_t elapsed = swGlobal.elapsed(StopWatch::Milliseconds);
            bool finish = swGlobal.elapsedFromRoot(StopWatch::Milliseconds) > finishTime;
            if (elapsed > nextTimeToOutput || finish) {
                auto output = (const RGB*)context->mapOutputBuffer();

                for (int y = 0; y < renderTargetSizeY; ++y) {
                    for (int x = 0; x < renderTargetSizeX; ++x) {
                        RGB srcPix = output[y * renderTargetSizeX + x];
                        uint32_t &pix = data[y * renderTargetSizeX + x];

                        srcPix *= g_brightnessCoeff;
                        srcPix = RGB::One() - exp(-srcPix);
                        srcPix = sRGB_gamma(srcPix);

                        pix = ((std::min<uint8_t>(srcPix.r * 256, 255) << 0) |
                               (std::min<uint8_t>(srcPix.g * 256, 255) << 8) |
                               (std::min<uint8_t>(srcPix.b * 256, 255) << 16) |
                               (0xFF << 24));
                    }
                }

                char filename[256];
                //sprintf(filename, "%03u.png", imgIndex++);
                //stbi_write_png(filename, renderTargetSizeX, renderTargetSizeY, 4, data, sizeof(data[0]) * renderTargetSizeX);
                sprintf(filename, "%03u.bmp", imgIndex++);
                stbi_write_bmp(filename, renderTargetSizeX, renderTargetSizeY, 4, data);
                vlrprintf("%u [spp]: %s, %g [s]\n", numAccumFrames, filename, elapsed * 1e-3f);

                context->unmapOutputBuffer();

                if (finish)
                    break;

                nextTimeToOutput += deltaTime;
                nextTimeToOutput = std::min(nextTimeToOutput, finishTime);
            }
        }
        delete[] data;

        swGlobal.stop();

        vlrprintf("Finish!!: %g[s]\n", swGlobal.stop(StopWatch::Milliseconds) * 1e-3f);
    }

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (optix::Exception ex) {
        vlrprintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
    }

    return 0;
}