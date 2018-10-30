# VLR: 

![VLR](README_TOP.png)  
IBL image: [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html)  

VLRはNVIDIA OptiXを使用したGPUモンテカルロレイトレーシングレンダラーです。  
VLR is a GPU Monte Carlo ray tracing renderer using NVIDIA OptiX.

## 特徴 / Features
* GPU Renderer using NVIDIA OptiX
* BSDFs
    * Ideal Diffuse (Lambert) BRDF
    * Ideal Specular BRDF/BSDF
    * Microfacet (GGX) BRDF/BSDF
    * Fresnel-blended Lambertian BSDF
    * UE4- or Frostbite-like BRDF \[Karis2013, Lagarde2014\]
    * Mixed BSDF
* Shader Node System
* Bump Mapping (Normal Map)
* Light Source Types
    * Area (Polygonal) Light
    * Infinitely Distant Image Based Environmental Light
* Camera Types
    * Perspective Camera with Depth of Field (thin-lens model)
    * Environment (Equirectangular) Camera
* Geometry Instancing
* Light Transport Algorithms
    * Path Tracing \[Kajiya1986\] with MIS
* Correct handling of non-symmetric scattering due to shading normals \[Veach1996, 1997\]

## 動作環境 / Confirmed Environment
現状以下の環境で動作を確認しています。  
I've confirmed that the program runs correctly on the following environment.

* Windows 10 & Visual Studio 2017 (15.8.5)
* MacBook Pro Retina Late 2013 (GT 750M 2GB)

動作させるにあたっては以下のライブラリが必要です。  
It requires the following libraries.

* CUDA 10.0
* OptiX 5.1
* OpenEXR 2.2
* assimp 4.1

## 注意 / Note
モデルデータやテクスチャーを読み込むシーンファイルがありますが、それらアセットはリポジトリには含まれていません。

There are some scene files loading model data and textures, but those assets are NOT included in this repository.

## 参考文献 / References
[Kajiya1986] "THE RENDERING EQUATION"  
[Karis2013] "Real Shading in Unreal Engine 4"  
[Lagarde2014] "Moving Frostbite to Physically Based Rendering 3.0"  
[Veach1996] "Non-symmetric Scattering in Light Transport Algorithms"  

----
2018 [@Shocker_0x15](https://twitter.com/Shocker_0x15)
