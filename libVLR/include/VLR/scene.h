#pragma once

#include "common.h"
#include "basic_types.h"

// shared_ptrもとりあえず使ってしまう。
// そもそもOptiXを使うのでVLRの場合はshared_ptrを使うことによるパフォーマンス上の懸念は特に無い。
// 一旦VLR内で各種データのメモリ領域を持つこととする。
// SLRSceneGraphの機能も内包した形で作っておく。

namespace VLR {
    class Context;

    class Transform;
    class StaticTransform;

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

    class Node;
    class SurfaceNode;
    struct Vertex;
    struct MaterialGroup;
    class TriangleMeshSurfaceNode;
    class InternalNode;
    class Scene;



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



    enum class DataFormat {
        RGB8x3 = 0,
        RGB_8x4,
        RGBA8x4,
        //RGBA16Fx4,
        Gray8,
        Num
    };



    struct Vertex {
        Point3D position;
        Normal3D normal;
        Vector3D tangent;
        TexCoord2D texCoord;
    };
}
