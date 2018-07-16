#pragma once

#include "common.h"
#include "basic_types.h"

// shared_ptrもとりあえず使ってしまう。
// そもそもOptiXを使うのでVLRの場合はshared_ptrを使うことによるパフォーマンス上の懸念は特に無い。
// 一旦VLR内で各種データのメモリ領域を持つこととする。
// SLRSceneGraphの機能も内包した形で作っておく。

namespace VLR {
    class Transform;
    class StaticTransform;

    class Node;
    class InternalNode;
    class SurfaceNode;
    struct Vertex;
    struct MaterialGroup;
    class TriangleMeshSurfaceNode;

    class Image2D;
    class LinearImage2D;

    class FloatTexture;
    class Float2Texture;
    class Float3Texture;
    class Float4Texture;
    class ImageFloat4Texture;

    class SurfaceMaterial;
    class MatteSurfaceMaterial;
    class UE4SurfaceMaterial;



    using TransformRef = std::shared_ptr<Transform>;

    using NodeRef = std::shared_ptr<Node>;
    using InternalNodeRef = std::shared_ptr<InternalNode>;
    using SurfaceNodeRef = std::shared_ptr<SurfaceNode>;

    using Image2DRef = std::shared_ptr<Image2D>;

    using FloatTextureRef = std::shared_ptr<FloatTexture>;
    using Float2TextureRef = std::shared_ptr<Float2Texture>;
    using Float3TextureRef = std::shared_ptr<Float3Texture>;
    using Float4TextureRef = std::shared_ptr<Float4Texture>;

    using SurfaceMaterialRef = std::shared_ptr<SurfaceMaterial>;



#define VLR_PIMPL_DECLARETION \
protected: \
    class Impl; \
private: \
    std::unique_ptr<Impl> m_privateImpl;



    class Transform {
    public:
        virtual ~Transform() {}

        virtual bool isStatic() const = 0;
    };



    class StaticTransform : public Transform {
        Matrix4x4 m_matrix;
        Matrix4x4 m_invMatrix;
    public:
        StaticTransform(const Matrix4x4 &m = Matrix4x4::Identity()) : m_matrix(m), m_invMatrix(invert(m)) {}

        bool isStatic() const override { return true; }

        StaticTransform operator*(const Matrix4x4 &m) const { return StaticTransform(m_matrix * m); }
        StaticTransform operator*(const StaticTransform &t) const { return StaticTransform(m_matrix * t.m_matrix); }
        bool operator==(const StaticTransform &t) const { return m_matrix == t.m_matrix; }
        bool operator!=(const StaticTransform &t) const { return m_matrix != t.m_matrix; }

        void getArrays(float mat[16], float invMat[16]) const {
            m_matrix.getArray(mat);
            m_invMatrix.getArray(invMat);
        }
    };



//    class VLR_API Node {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Node();
//        virtual ~Node();
//    };
//
//
//
//    class VLR_API InternalNode : public Node {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        InternalNode(const TransformRef &localToWorld);
//        ~InternalNode();
//
//        void addChild(NodeRef node);
//        
//        void setTransform(const TransformRef &localToWorld);
//    };
//
//
//
//    class VLR_API SurfaceNode : public Node {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        SurfaceNode();
//        ~SurfaceNode();
//    };
//
//
//
//    struct VLR_API Vertex {
//        Point3D position;
//        Normal3D normal;
//        Vector3D tangent;
//        TexCoord2D texCoord;
//    };
//
//    struct VLR_API MaterialGroup {
//        std::vector<uint32_t> indices;
//        SurfaceMaterialRef material;
//    };
//
//    class VLR_API TriangleMeshSurfaceNode : public SurfaceNode {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        TriangleMeshSurfaceNode();
//        ~TriangleMeshSurfaceNode();
//
//        void addVertices(const Vertex* vertices, uint32_t numVertices);
//    };



    enum class DataFormat {
        RGB8x3 = 0,
        RGB_8x4,
        RGBA8x4,
        //RGBA16Fx4,
        Gray8,
        Num
    };

    struct RGB8x3 { uint8_t r, g, b; };
    struct RGB_8x4 { uint8_t r, g, b, dummy; };
    struct RGBA8x4 { uint8_t r, g, b, a; };
    //struct RGBA16Fx4 { half r, g, b, a; };
    struct Gray8 { uint8_t v; };

    extern VLR_API const size_t sizesOfDataFormats[(uint32_t)DataFormat::Num];
    


//    class VLR_API Image2D {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Image2D(uint32_t width, uint32_t height, DataFormat dataFormat);
//        virtual ~Image2D();
//
//        uint32_t getWidth() const;
//        uint32_t getHeight() const;
//        uint32_t getStride() const;
//
//        static DataFormat getInternalFormat(DataFormat inputFormat);
//    };
//
//
//
//    class VLR_API LinearImage2D : public Image2D {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        LinearImage2D(const uint8_t* linearData, uint32_t width, uint32_t height, DataFormat dataFormat);
//        ~LinearImage2D();
//    };
//
//
//
//    class VLR_API FloatTexture {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        FloatTexture();
//        virtual ~FloatTexture();
//    };
//
//
//
//    class VLR_API Float2Texture {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Float2Texture();
//        virtual ~Float2Texture();
//    };
//
//
//
//    class VLR_API Float3Texture {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Float3Texture();
//        virtual ~Float3Texture();
//    };
//
//
//
//    class VLR_API Float4Texture {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Float4Texture();
//        virtual ~Float4Texture();
//    };
//
//
//
//    class VLR_API ImageFloat4Texture : public Float4Texture {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        ImageFloat4Texture();
//        ~ImageFloat4Texture();
//    };
//
//
//
//    class VLR_API SurfaceMaterial {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        SurfaceMaterial();
//        virtual ~SurfaceMaterial();
//    };
//
//
//
//    class VLR_API MatteSurfaceMaterial : public SurfaceMaterial {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        MatteSurfaceMaterial(const Float4TextureRef &albedoRoughnessTex);
//        ~MatteSurfaceMaterial();
//    };
//
//
//
//    class VLR_API UE4SurfaceMaterial : public SurfaceMaterial {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        UE4SurfaceMaterial(const Float3TextureRef &baseColorTex, const Float2TextureRef &roughnessMetallicTex);
//        ~UE4SurfaceMaterial();
//    };
//
//
//
//    class VLR_API Scene {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Scene();
//        ~Scene();
//
//        void addChild(NodeRef node);
//
//        void setTransform(const TransformRef &localToWorld);
//    };
//
//
//
//    class VLR_API Context {
//        VLR_PIMPL_DECLARETION
//
//    public:
//        Context();
//        ~Context();
//    };
#undef VLR_PIMPL_DECLARETION
}
