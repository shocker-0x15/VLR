#include <cstdio>
#include <cstdint>

#include <VLR/VLR.h>

#include <optix_world.h>

#define NOMINMAX
#include "imgui.h"
#include "imgui_impl_glfw_gl3.h"
#include "GLToolkit.h"
#include "GLFW/glfw3.h"

// DELETE ME
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#ifdef DEBUG
#   define ENABLE_ASSERT
#endif

#ifdef VLR_Platform_Windows_MSVC
static void debugPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[1024];
    vsprintf_s(str, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#else
#   define debugPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) if (!(expr)) { debugPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); debugPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")

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
KeyState g_buttonRotate;
double g_mouseX;
double g_mouseY;

VLR::Point3D g_cameraPos;
VLR::Quaternion g_cameraOrientation;
float g_lensRadius;
float g_objPlaneDistance;

static std::string readTxtFile(const std::string& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
};

static void glfw_error_callback(int32_t error, const char* description) {
    debugPrintf("Error %d: %s\n", error, description);
}

static VLRCpp::Image2DRef loadImage2D(VLRCpp::Context &context, const std::string &filepath) {
    using namespace VLRCpp;
    using namespace VLR;

    Image2DRef ret;

    int32_t width, height, n;
    uint8_t* imageData = stbi_load(filepath.c_str(), &width, &height, &n, 4);
    ret = context.createLinearImage2D(width, height, DataFormat::RGBA8x4, imageData);
    stbi_image_free(imageData);

    return ret;
}

static void recursiveConstruct(VLRCpp::Context &context, const aiScene* objSrc, const aiNode* nodeSrc,
                               const std::vector<VLRCpp::SurfaceMaterialRef> &materials, /*const std::vector<NormalTextureRef> &normalMaps, const std::vector<FloatTextureRef> &alphaMaps,*/
                               bool flipV,
                               VLRCpp::InternalNodeRef* nodeOut) {
    using namespace VLRCpp;
    using namespace VLR;

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

    *nodeOut = context.createInternalNode(nodeSrc->mName.C_Str(), createShared<StaticTransform>(Matrix4x4(tfElems)));

    std::vector<uint32_t> meshIndices;
    for (int m = 0; m < nodeSrc->mNumMeshes; ++m) {
        const aiMesh* mesh = objSrc->mMeshes[nodeSrc->mMeshes[m]];
        if (mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            debugPrintf("ignored non triangle mesh.\n");
            continue;
        }

        auto surfMesh = context.createTriangleMeshSurfaceNode(mesh->mName.C_Str());
        const SurfaceMaterialRef &surfMat = materials[mesh->mMaterialIndex];
        //const NormalTextureRef &normalMap = normalMaps[mesh->mMaterialIndex];
        //const FloatTextureRef &alphaMap = alphaMaps[mesh->mMaterialIndex];

        std::vector<Vertex> vertices;
        for (int v = 0; v < mesh->mNumVertices; ++v) {
            const aiVector3D &p = mesh->mVertices[v];
            const aiVector3D &n = mesh->mNormals[v];
            Vector3D tangent, bitangent;
            if (mesh->mTangents == nullptr)
                Normal3D(n.x, n.y, n.z).makeCoordinateSystem(&tangent, &bitangent);
            const aiVector3D &t = mesh->mTangents ? mesh->mTangents[v] : aiVector3D(tangent[0], tangent[1], tangent[2]);
            const aiVector3D &uv = mesh->mNumUVComponents[0] > 0 ? mesh->mTextureCoords[0][v] : aiVector3D(0, 0, 0);

            Vertex outVtx{ Point3D(p.x, p.y, p.z), Normal3D(n.x, n.y, n.z), Vector3D(t.x, t.y, t.z), TexCoord2D(uv.x, flipV ? (1 - uv.y) : uv.y) };
            float dotNT = dot(outVtx.normal, outVtx.tangent);
            if (std::fabs(dotNT) >= 0.01f)
                outVtx.tangent = normalize(outVtx.tangent - dotNT * outVtx.normal);
            //SLRAssert(absDot(outVtx.normal, outVtx.tangent) < 0.01f, "shading normal and tangent must be orthogonal: %g", absDot(outVtx.normal, outVtx.tangent));
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
        surfMesh->addMaterialGroup(meshIndices.data(), meshIndices.size(), surfMat);

        (*nodeOut)->addChild(surfMesh);
    }

    if (nodeSrc->mNumChildren) {
        for (int c = 0; c < nodeSrc->mNumChildren; ++c) {
            InternalNodeRef subNode;
            recursiveConstruct(context, objSrc, nodeSrc->mChildren[c], materials, /*normalMaps, alphaMaps, */flipV, &subNode);
            if (subNode != nullptr)
                (*nodeOut)->addChild(subNode);
        }
    }
}

struct SurfaceAttributeTuple {
    VLRCpp::SurfaceMaterialRef material;

    SurfaceAttributeTuple(const VLRCpp::SurfaceMaterialRef &_material) : 
        material(_material) {}
};

static SurfaceAttributeTuple createMaterialDefaultFunction(VLRCpp::Context &context, const aiMaterial* aiMat, const std::string &pathPrefix) {
    using namespace VLRCpp;
    using namespace VLR;

    aiReturn ret;
    (void)ret;
    aiString strValue;
    float color[3];

    aiMat->Get(AI_MATKEY_NAME, strValue);

    Float4TextureRef texAlbedoRoughness;
    if (aiMat->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), strValue) == aiReturn_SUCCESS) {
        Image2DRef image = loadImage2D(context, pathPrefix + strValue.C_Str());
        texAlbedoRoughness = context.createImageFloat4Texture(image);
    }
    else if (aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color, nullptr) == aiReturn_SUCCESS) {
        float value[4] = { color[0], color[1], color[2], 0.0f };
        texAlbedoRoughness = context.createConstantFloat4Texture(value);
    }
    else {
        float value[4] = { 1.0f, 0.0f, 1.0f, 0.0f };
        texAlbedoRoughness = context.createConstantFloat4Texture(value);
    }

    SurfaceMaterialRef mat = context.createMatteSurfaceMaterial(texAlbedoRoughness);

    return SurfaceAttributeTuple(mat);
}

static void construct(VLRCpp::Context &context, const std::string &filePath, bool flipV, VLRCpp::InternalNodeRef* nodeOut) {
    using namespace VLRCpp;
    using namespace VLR;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath, 0);
    if (!scene) {
        debugPrintf("Failed to load %s.\n", filePath.c_str());
        return;
    }
    debugPrintf("Reading: %s done.\n", filePath.c_str());

    std::string pathPrefix = filePath.substr(0, filePath.find_last_of("/") + 1);

    // create materials
    std::vector<SurfaceMaterialRef> materials;
    //std::vector<NormalTextureRef> normalMaps;
    //std::vector<FloatTextureRef> alphaMaps;
    for (int m = 0; m < scene->mNumMaterials; ++m) {
        const aiMaterial* aiMat = scene->mMaterials[m];

        SurfaceAttributeTuple surfAttr = createMaterialDefaultFunction(context, aiMat, pathPrefix);
        materials.push_back(surfAttr.material);
        //normalMaps.push_back(surfAttr.normalMap);
        //alphaMaps.push_back(surfAttr.alphaMap);
    }

    recursiveConstruct(context, scene, scene->mRootNode, materials, /*normalMaps, alphaMaps, */flipV, nodeOut);

    debugPrintf("Constructing: %s done.\n", filePath.c_str());
}



static int32_t mainFunc(int32_t argc, const char* argv[]) {
    using namespace VLRCpp;
    using namespace VLR;

    VLRCpp::Context context;

    SceneRef scene = context.createScene(std::make_shared<StaticTransform>(translate(0.0f, 0.0f, 0.0f)));

    InternalNodeRef modelNode;
    construct(context, "resources/Kirby_Pikachu_Hat/pikachu_hat_corrected.obj", true, &modelNode);
    scene->addChild(modelNode);

    TriangleMeshSurfaceNodeRef cornellBox = context.createTriangleMeshSurfaceNode("CornellBox");
    {
        std::vector<Vertex> vertices;
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D( 0,  1, 0), Vector3D( 1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  0.0f, -1.5f), Normal3D( 0,  1, 0), Vector3D( 1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  0.0f,  1.5f), Normal3D( 0,  1, 0), Vector3D( 1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f,  1.5f), Normal3D( 0,  1, 0), Vector3D( 1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D( 0,  0, 1), Vector3D( 1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  0.0f, -1.5f), Normal3D( 0,  0, 1), Vector3D( 1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  3.0f, -1.5f), Normal3D( 0,  0, 1), Vector3D( 1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f, -1.5f), Normal3D( 0,  0, 1), Vector3D( 1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f, -1.5f), Normal3D( 0, -1, 0), Vector3D( 1,  0,  0), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  3.0f, -1.5f), Normal3D( 0, -1, 0), Vector3D( 1,  0,  0), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  3.0f,  1.5f), Normal3D( 0, -1, 0), Vector3D( 1,  0,  0), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f,  1.5f), Normal3D( 0, -1, 0), Vector3D( 1,  0,  0), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f,  1.5f), Normal3D( 1,  0, 0), Vector3D( 0,  0, -1), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  0.0f, -1.5f), Normal3D( 1,  0, 0), Vector3D( 0,  0, -1), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f, -1.5f), Normal3D( 1,  0, 0), Vector3D( 0,  0, -1), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D(-1.5f,  3.0f,  1.5f), Normal3D( 1,  0, 0), Vector3D( 0,  0, -1), TexCoord2D(0.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  0.0f, -1.5f), Normal3D(-1,  0, 0), Vector3D( 0,  0,  1), TexCoord2D(0.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  0.0f,  1.5f), Normal3D(-1,  0, 0), Vector3D( 0,  0,  1), TexCoord2D(1.0f, 1.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  3.0f,  1.5f), Normal3D(-1,  0, 0), Vector3D( 0,  0,  1), TexCoord2D(1.0f, 0.0f) });
        vertices.push_back(Vertex{ Point3D( 1.5f,  3.0f, -1.5f), Normal3D(-1,  0, 0), Vector3D( 0,  0,  1), TexCoord2D(0.0f, 0.0f) });
        cornellBox->setVertices(vertices.data(), vertices.size());

        {
            float value[4] = { 0.75f, 0.75f, 0.75f, 0.0f };
            Float4TextureRef texAlbedoRoughness = context.createConstantFloat4Texture(value);
            SurfaceMaterialRef matMatte = context.createMatteSurfaceMaterial(texAlbedoRoughness);

            std::vector<uint32_t> matGroup = {
                0, 1, 2, 0, 2, 3,
                4, 5, 6, 4, 6, 7,
                8, 9, 10, 8, 10, 11,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte);
        }

        {
            float value[4] = { 0.75f, 0.25f, 0.25f, 0.0f };
            Float4TextureRef texAlbedoRoughness = context.createConstantFloat4Texture(value);
            SurfaceMaterialRef matMatte = context.createMatteSurfaceMaterial(texAlbedoRoughness);

            std::vector<uint32_t> matGroup = {
                12, 13, 14, 12, 14, 15,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte);
        }

        {
            float value[4] = { 0.25f, 0.25f, 0.75f, 0.0f };
            Float4TextureRef texAlbedoRoughness = context.createConstantFloat4Texture(value);
            SurfaceMaterialRef matMatte = context.createMatteSurfaceMaterial(texAlbedoRoughness);

            std::vector<uint32_t> matGroup = {
                16, 17, 18, 16, 18, 19,
            };
            cornellBox->addMaterialGroup(matGroup.data(), matGroup.size(), matMatte);
        }
    }

    scene->addChild(cornellBox);

    g_cameraPos = Point3D(0, 0, 5);
    g_cameraOrientation = qRotateY<float>(M_PI);
    g_lensRadius = 0.0f;
    g_objPlaneDistance = 1.0f;
    PerspectiveCameraRef camera = context.createPerspectiveCamera(g_cameraPos, g_cameraOrientation, 
                                                                  1280.0f / 720.0f, 40 * M_PI / 180, g_lensRadius, 1.0f, g_objPlaneDistance);



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
    const uint32_t WindowSizeX = 1280; // not in pixels
    const uint32_t WindowSizeY = 720;
    float contentScaleX, contentScaleY;
    glfwGetMonitorContentScale(primaryMonitor, &contentScaleX, &contentScaleY);
    const float UIScaling = contentScaleX;
    GLFWwindow* window = glfwCreateWindow((int32_t)(WindowSizeX * UIScaling), (int32_t)(WindowSizeY * UIScaling), "VLR", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync

                         // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        glfwTerminate();
        debugPrintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
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
    outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(VLR::RGBSpectrum), WindowSizeX * WindowSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);

    context.bindOpenGLBuffer(outputBufferGL.getRawHandle(), WindowSizeX, WindowSizeY);

    GLTK::BufferTexture outputTexture;
    outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGB32F);

    // JP: OptiXの出力を書き出すシェーダー。
    GLTK::GraphicsShader drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(readTxtFile("resources/shaders/drawOptiXResult.vert"),
                                         readTxtFile("resources/shaders/drawOptiXResult.frag"));

    // JP: HiDPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
    GLTK::FrameBuffer frameBuffer;
    frameBuffer.initialize(WindowSizeX, WindowSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

    // JP: アップスケール用のシェーダー。
    GLTK::GraphicsShader scaleShader;
    scaleShader.initializeVSPS(readTxtFile("resources/shaders/scale.vert"),
                               readTxtFile("resources/shaders/scale.frag"));

    // JP: アップスケール用のサンプラー。
    //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
    GLTK::Sampler scaleSampler;
    scaleSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest, GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
        switch (button) {
        case GLFW_MOUSE_BUTTON_MIDDLE: {
            debugPrintf("Mouse Middle\n");
            g_buttonRotate.recordStateChange(action == GLFW_PRESS, g_frameIndex);
            break;
        }
        default:
            break;
        }
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        g_mouseX = x;
        g_mouseY = y;
    });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
        switch (key) {
        case GLFW_KEY_W: {
            debugPrintf("W: %d\n", action);
            g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_S: {
            debugPrintf("S: %d\n", action);
            g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_A: {
            debugPrintf("A: %d\n", action);
            g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_D: {
            debugPrintf("D: %d\n", action);
            g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_R: {
            debugPrintf("R: %d\n", action);
            g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_F: {
            debugPrintf("F: %d\n", action);
            g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_Q: {
            debugPrintf("Q: %d\n", action);
            g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        case GLFW_KEY_E: {
            debugPrintf("E: %d\n", action);
            g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, g_frameIndex);
            break;
        }
        default:
            break;
        }
    });

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        bool cameraIsMoving = false;

        // process key events
        {
            int32_t trackZ = 0;
            if (g_keyForward.getState() == true) {
                if (g_keyBackward.getState() == true)
                    trackZ = 0;
                else
                    trackZ = 1;
            }
            else {
                if (g_keyBackward.getState() == true)
                    trackZ = -1;
                else
                    trackZ = 0;
            }

            int32_t trackX = 0;
            if (g_keyLeftward.getState() == true) {
                if (g_keyRightward.getState() == true)
                    trackX = 0;
                else
                    trackX = 1;
            }
            else {
                if (g_keyRightward.getState() == true)
                    trackX = -1;
                else
                    trackX = 0;
            }

            int32_t trackY = 0;
            if (g_keyUpward.getState() == true) {
                if (g_keyDownward.getState() == true)
                    trackY = 0;
                else
                    trackY = 1;
            }
            else {
                if (g_keyDownward.getState() == true)
                    trackY = -1;
                else
                    trackY = 0;
            }

            int32_t tiltZ = 0;
            if (g_keyTiltRight.getState() == true) {
                if (g_keyTiltLeft.getState() == true)
                    tiltZ = 0;
                else
                    tiltZ = 1;
            }
            else {
                if (g_keyTiltLeft.getState() == true)
                    tiltZ = -1;
                else
                    tiltZ = 0;
            }

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
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

            g_cameraPos += g_cameraOrientation.toMatrix3x3() * 0.05f * Vector3D(trackX, trackY, trackZ);
            g_cameraOrientation = g_cameraOrientation * qRotateZ(0.025f * tiltZ);
            Quaternion tempOrientation = g_cameraOrientation * qRotate(0.15f * 1e-2f * deltaAngle, axis);
            if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == g_frameIndex) {
                g_cameraOrientation = tempOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            camera->setPosition(g_cameraPos);
            camera->setOrientation(tempOrientation);
            camera->setLensRadius(g_lensRadius);
            camera->setObjectPlaneDistance(g_objPlaneDistance);

            cameraIsMoving = trackZ != 0 || trackX != 0 || trackY != 0 || tiltZ != 0 || deltaX != 0 || deltaY != 0;
        }

        {
            uint32_t shrinkCoeff = cameraIsMoving ? 4 : 1;

            context.render(scene, camera, shrinkCoeff);

            ImGui_ImplGlfwGL3_NewFrame(WindowSizeX, WindowSizeY, UIScaling);

            {
                ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

                ImGui::InputFloat3("Position", (float*)&g_cameraPos);

                ImGui::SliderFloat("Lens Radius", &g_lensRadius, 0.0f, 0.15f, "%.3f", 1.0f);
                ImGui::SliderFloat("Object Plane Distance", &g_objPlaneDistance, 0.01f, 20.0f, "%.3f", 2.0f);

                ImGui::End();
            }

            {
                ImGui::Begin("Scene");

                ImGui::End();
            }

            frameBuffer.bind(GLTK::FrameBuffer::Target::ReadDraw);

            glViewport(0, 0, frameBuffer.getWidth(), frameBuffer.getHeight());
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClearDepth(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            {
                drawOptiXResultShader.useProgram();

                glUniform1i(0, (int32_t)WindowSizeX); GLTK::errorCheck();

                glUniform1f(1, (float)shrinkCoeff); GLTK::errorCheck();

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
        }

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

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (optix::Exception ex) {
        VLRDebugPrintf("OptiX Error: %u: %s\n", ex.getErrorCode(), ex.getErrorString().c_str());
    }

    return 0;
}