#include "app.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <random>
#include <chrono>

#include <comdef.h>
#include <d3d11.h>

#pragma warning(disable: 4201)

_COM_SMARTPTR_TYPEDEF(ID3D11Device, __uuidof(ID3D11Device));
_COM_SMARTPTR_TYPEDEF(IDXGISwapChain, __uuidof(IDXGISwapChain));
_COM_SMARTPTR_TYPEDEF(ID3D11DeviceContext, __uuidof(ID3D11DeviceContext));
_COM_SMARTPTR_TYPEDEF(ID3D11Texture2D, __uuidof(ID3D11Texture2D));
_COM_SMARTPTR_TYPEDEF(ID3D11RenderTargetView, __uuidof(ID3D11RenderTargetView));
_COM_SMARTPTR_TYPEDEF(ID3D11VertexShader, __uuidof(ID3D11VertexShader));
_COM_SMARTPTR_TYPEDEF(ID3D11PixelShader, __uuidof(ID3D11PixelShader));
_COM_SMARTPTR_TYPEDEF(ID3D11Buffer, __uuidof(ID3D11Buffer));
_COM_SMARTPTR_TYPEDEF(ID3D11InputLayout, __uuidof(ID3D11InputLayout));
_COM_SMARTPTR_TYPEDEF(ID3D11RasterizerState, __uuidof(ID3D11RasterizerState));
_COM_SMARTPTR_TYPEDEF(ID3D11DepthStencilState, __uuidof(ID3D11DepthStencilState));
_COM_SMARTPTR_TYPEDEF(ID3D11ShaderResourceView, __uuidof(ID3D11ShaderResourceView));
_COM_SMARTPTR_TYPEDEF(ID3D11SamplerState, __uuidof(ID3D11SamplerState));


using namespace std;
using namespace std::chrono;

namespace
{

template<typename Func>
void ThrowOnFailure(Func f, const char* expr) {
	HRESULT hr = f();
	if (FAILED(hr))
		throw exception(expr);
}

#define THROW_ON_FAILURE(exp) (ThrowOnFailure([&]() { return (exp); }, #exp))

vector<char> LoadFile(const char* filename) {
	ifstream file(filename, ios_base::binary);
	return vector<char>(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
}

ID3D11VertexShaderPtr LoadVertexShader(ID3D11Device* device, const char* filename) {
	const auto buf = LoadFile(filename);
	ID3D11VertexShaderPtr vertexShader;
	THROW_ON_FAILURE(device->CreateVertexShader(&buf[0], buf.size(), NULL, &vertexShader));
	return vertexShader;
}

ID3D11PixelShaderPtr LoadPixelShader(ID3D11Device* device, const char* filename) {
	const auto buf = LoadFile(filename);
	ID3D11PixelShaderPtr pixelShader;
	THROW_ON_FAILURE(device->CreatePixelShader(&buf[0], buf.size(), NULL, &pixelShader));
	return pixelShader;
}

struct Vertex {
	float x, y, z;
};

struct Vector2 {
	Vector2() {}
	Vector2(float x_, float y_) : x(x_), y(y_) {}
	float x, y;
};

inline Vector2 operator+(const Vector2& v, const float x) {
	return Vector2(v.x + x, v.y + x);
}

inline Vector2 operator*(const Vector2& v, const float x) {
	return Vector2(v.x * x, v.y * x);
}

struct Vector3 {
	Vector3() {}
	explicit Vector3(float x_) : x(x_), y(x_), z(x_) {}
	Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
	float x, y, z;
};

inline Vector3 operator+(const Vector3& v, const float x) {
	return Vector3(v.x + x, v.y + x, v.z + x);
}

inline Vector3 operator*(const Vector3& v, const float x) {
	return Vector3(v.x * x, v.y * x, v.z * x);
}

struct PSConstants {
	Vector2 screenSize;
	Vector2 texSize;
};

struct VSConstants {
	Vector2 offset;
	Vector2 pad;
};

template<typename T>
T Clamp(T x, T a, T b) {
	return max(min(x, b), a);
}

Vector3 Clamp(Vector3 v, float a, float b) {
	return Vector3(Clamp(v.x, a, b), Clamp(v.y, a, b), Clamp(v.z, a, b));
}

struct ColorR8G8B8A8 {
	union {
		uint32_t rgba;
		struct { uint8_t r, g, b, a; };
	};
};

inline ColorR8G8B8A8 MakeColorR8G8B8A8(float r, float g, float b, float a) {
	const float gamma = 1.0f / 2.0f;
	ColorR8G8B8A8 col;
	col.r = static_cast<uint8_t>(pow(Clamp(r, 0.0f, 1.0f), gamma) * 255.0f);
	col.g = static_cast<uint8_t>(pow(Clamp(g, 0.0f, 1.0f), gamma) * 255.0f);
	col.b = static_cast<uint8_t>(pow(Clamp(b, 0.0f, 1.0f), gamma) * 255.0f);
	col.a = static_cast<uint8_t>(Clamp(a * 255.0f, 0.0f, 255.0f));
	return col;
}

inline ColorR8G8B8A8 MakeColorR8G8B8A8(const Vector3& col) {
	return MakeColorR8G8B8A8(col.x, col.y, col.z, 1.0f);
}

template<typename VertexType>
ID3D11BufferPtr CreateVertexBuffer(ID3D11Device* device, const VertexType* vertices, size_t numVertices) {
	CD3D11_BUFFER_DESC desc(static_cast<UINT>(sizeof(VertexType) * numVertices), 
		D3D11_BIND_VERTEX_BUFFER, D3D11_USAGE_IMMUTABLE);

	D3D11_SUBRESOURCE_DATA data = { vertices };
	
	ID3D11BufferPtr vertexBuffer;
	THROW_ON_FAILURE(device->CreateBuffer(&desc, &data, &vertexBuffer));
	return vertexBuffer;
}

template<typename StructType>
ID3D11BufferPtr CreateConstantBuffer(ID3D11Device* device, const StructType* constants) {
	CD3D11_BUFFER_DESC desc(static_cast<UINT>(sizeof(StructType)), D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DEFAULT);
	D3D11_SUBRESOURCE_DATA data = { constants };
	ID3D11BufferPtr constantBuffer;
	THROW_ON_FAILURE(device->CreateBuffer(&desc, &data, &constantBuffer));
	return constantBuffer;
}

ID3D11InputLayoutPtr CreateInputLayout(ID3D11Device* device, const vector<char>& shaderBytecode) {
	ID3D11InputLayoutPtr inputLayout;
	const int numElements = 1;
	D3D11_INPUT_ELEMENT_DESC desc[numElements] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};
	THROW_ON_FAILURE(device->CreateInputLayout(&desc[0], numElements, &shaderBytecode[0], shaderBytecode.size(), &inputLayout));
	return inputLayout;
}

float Fade(float t) {
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

template<typename T>
float Lerp(float t, T a, T b) {
	return a + t * (b - a);
}

template<typename T>
float Bilerp(float s, float t, T v00, T v10, T v01, T v11) {
	return Lerp(t, Lerp(s, v00, v10), Lerp(s, v01, v11));
}

float Grad(int hash, float x, float y, float z) {
	auto h = hash & 15;
	auto u = h < 8 ? x : y;
	auto v = h < 4 ? y : (h == 12 || h == 14) ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

float Noise(float x, float y, float z) {
	static int p[] = { 151,160,137,91,90,15,
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
		190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
		88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
		77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
		102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
		135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
		5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
		223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
		129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
		251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
		49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
		151,160,137,91,90,15,
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
		190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
		88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
		77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
		102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
		135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
		5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
		223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
		129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
		251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
		49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180	
	};

	int xi = static_cast<int>(floor(x)) & 0xff;
	int yi = static_cast<int>(floor(y)) & 0xff;
	int zi = static_cast<int>(floor(z)) & 0xff;

	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	auto u = Fade(x);
	auto v = Fade(y);
	auto w = Fade(z);

	auto a = p[xi] + yi;
	auto aa = p[a] + zi;
	auto ab = p[a+1] + zi;
	auto b = p[xi+1] + yi;
	auto ba = p[b] + zi;
	auto bb = p[b+1] + zi;

	return Lerp(w, Lerp(v,	Lerp(u, Grad(p[aa  ], x  , y  , z   ),  
                                    Grad(p[ba  ], x-1, y  , z   )), 
                            Lerp(u, Grad(p[ab  ], x  , y-1, z   ),  
                                    Grad(p[bb  ], x-1, y-1, z   ))),
                    Lerp(v, Lerp(u, Grad(p[aa+1], x  , y  , z-1 ),  
                                    Grad(p[ba+1], x-1, y  , z-1 )), 
                            Lerp(u, Grad(p[ab+1], x  , y-1, z-1 ),
                                    Grad(p[bb+1], x-1, y-1, z-1 ))));
}

float Turbulence(float x, float y, float z) {
	const float base = 2.0137f;
	float frac = 1.0f;
	float res = 0.0f;
	for (int i = 0; i < 6; ++i) {
		res += Noise(x, y, z) * frac;
		x *= base;
		y *= base;
		z *= base;
		frac /= base;
	}
	return res;
}

Vector3 Potential(float x, float y, float z) {
	return Vector3(Noise(x, y, z), Noise(x + 5.0f, y + 5.0f, z + 5.0f), 
		Noise(x + 10.0f, y + 10.0f, z + 10.0f));
}

Vector3 CurlNoise(float x, float y, float z) {
	const float delta = 1e-4f;
	Vector3 v;
	v.x=( (Potential(x, y+delta, z).z - Potential(x, y-delta, z).z)
		-(Potential(x, y, z+delta).y - Potential(x, y, z-delta).y) ) / (2*delta);
	v.y=( (Potential(x, y, z+delta).x - Potential(x, y, z-delta).x)
		-(Potential(x+delta, y, z).z - Potential(x-delta, y, z).z) ) / (2*delta);
	v.z=( (Potential(x+delta, y, z).y - Potential(x-delta, y, z).y)
		-(Potential(x, y+delta, z).x - Potential(x, y-delta, z).x) ) / (2*delta);
	return v;
}

template<typename T>
inline T Get(int x, int y, const T* data, size_t width) {
	return data[y * width + x];
}

template<typename T>
inline void Set(int x, int y, T* data, size_t width, T val) {
	data[y * width + x] = val;
}

void SetBoundContinuity(int n, float* x) {
	const int width = n + 2;
	for (int i = 1; i <= n; ++i) {
		Set(0, i, x, width, Get(1, i, x, width));
		Set(n + 1, i, x, width, Get(n, i, x, width));
		Set(i, 0, x, width, Get(i, 1, x, width));
		Set(i, n + 1, x, width, Get(i, n, x, width));
	}
	Set(0, 0, x, width, 0.5f * (Get(1, 0, x, width) + Get(0, 1, x, width)));
	Set(0, n + 1, x, width, 0.5f * (Get(1, n + 1, x, width) + Get(0, n, x, width)));
	Set(n + 1, 0, x, width, 0.5f * (Get(n, 0, x, width) + Get(n + 1, 1, x, width)));
	Set(n + 1, n + 1, x, width, 0.5f * (Get(n, n + 1, x, width) + Get(n + 1, n, x, width)));
}

void SetBoundTangentU(int n, float* x) {
	const int width = n + 2;
	for (int i = 1; i <= n; ++i) {
		Set(0, i, x, width, -Get(1, i, x, width));
		Set(n + 1, i, x, width, -Get(n, i, x, width));
		Set(i, 0, x, width, Get(i, 1, x, width));
		Set(i, n + 1, x, width, Get(i, n, x, width));
	}
	Set(0, 0, x, width, 0.5f * (Get(1, 0, x, width) + Get(0, 1, x, width)));
	Set(0, n + 1, x, width, 0.5f * (Get(1, n + 1, x, width) + Get(0, n, x, width)));
	Set(n + 1, 0, x, width, 0.5f * (Get(n, 0, x, width) + Get(n + 1, 1, x, width)));
	Set(n + 1, n + 1, x, width, 0.5f * (Get(n, n + 1, x, width) + Get(n + 1, n, x, width)));
}

void SetBoundTangentV(int n, float* x) {
	const int width = n + 2;
	for (int i = 1; i <= n; ++i) {
		Set(0, i, x, width, Get(1, i, x, width));
		Set(n + 1, i, x, width, Get(n, i, x, width));
		Set(i, 0, x, width, -Get(i, 1, x, width));
		Set(i, n + 1, x, width, -Get(i, n, x, width));
	}
	Set(0, 0, x, width, 0.5f * (Get(1, 0, x, width) + Get(0, 1, x, width)));
	Set(0, n + 1, x, width, 0.5f * (Get(1, n + 1, x, width) + Get(0, n, x, width)));
	Set(n + 1, 0, x, width, 0.5f * (Get(n, 0, x, width) + Get(n + 1, 1, x, width)));
	Set(n + 1, n + 1, x, width, 0.5f * (Get(n, n + 1, x, width) + Get(n + 1, n, x, width)));
}

void AddSource(size_t n, float* x, const float* x0, float dt) {
	const auto size = (n + 2) * (n + 2);
	for (auto i = 0; i < size; ++i) {
		x[i] += x0[i] * dt;
	}
}

void Relax(size_t n, float* x, const float* x0, const float a, const float b, const int bounds = 0) {
	const size_t width = n + 2;
	for (auto k = 0; k < 20; ++k) {
		for (auto j = 1; j <= n; ++j) {
			for (auto i = 1; i <= n; ++i) {
				const float val = (Get(i, j, x0, width) + a * (Get(i - 1, j, x, width) + Get(i + 1, j, x, width) 
					+ Get(i, j - 1, x, width) + Get(i, j + 1, x, width))) / (b + 4.0f * a);
				Set(i, j, x, width, val);
			}
		}
		switch (bounds) {
		case 0:
			SetBoundContinuity(static_cast<int>(n), x);
			break;
		case 1:
			SetBoundTangentU(static_cast<int>(n), x);
			break;
		case 2:
			SetBoundTangentV(static_cast<int>(n), x);
			break;
		}
	}
}

void Diffuse(size_t n, float* x, const float* x0, float diff, float dt, const int bounds = 0) {
	const float a = dt * diff * n * n;
	Relax(n, x, x0, a, 1.0f, bounds);
}

void Advect(const size_t n, float* d, const float* d0, const float* u, const float* v, const float dt, const int bounds = 0) {
	float dt0 = dt * n;
	for (auto j = 1; j <= n; ++j) {
		for (auto i = 1; i <= n; ++i) {
			const size_t width = n + 2;
			const auto x = Clamp(i - dt0 * Get(i, j, u, width), 0.5f, n + 0.5f);
			const auto y = Clamp(j - dt0 * Get(i, j, v, width), 0.5f, n + 0.5f);
			const auto i0 = static_cast<int>(x);
			const auto i1 = i0 + 1;
			const auto j0 = static_cast<int>(y);
			const auto j1 = j0 + 1;
			Set(i, j, d, width, Bilerp(x - i0, y - j0, Get(i0, j0, d0, width), Get(i0, j1, d0, width),
				Get(i1, j0, d0, width), Get(i1, j1, d0, width)));
		}
	}
	switch (bounds) {
	case 0:
		SetBoundContinuity(static_cast<int>(n), d);
		break;
	case 1:
		SetBoundTangentU(static_cast<int>(n), d);
		break;
	case 2:
		SetBoundTangentV(static_cast<int>(n), d);
		break;
	}
}

void Project(size_t n, float* u, float* v, float* p, float* div)
{
	const float h = 1.0f / n;
	const size_t width = n + 2;

	for (auto j = 1; j <= n; ++j) {
		for (auto i = 1; i <= n; ++i) {
			const float val = -0.5f * h * (Get(i + 1, j, u, width) - Get(i - 1, j, u, width)
										+ Get(i, j + 1, v, width) - Get(i, j - 1, v, width));
			Set(i, j, div, width, val);
			Set(i, j, p, width, 0.0f);
		}
	}

	SetBoundContinuity(static_cast<int>(n), div);
	SetBoundContinuity(static_cast<int>(n), p);

	Relax(n, p, div, 1.0f, 0.0f);

	for (auto j = 1; j <= n; ++j) {
		for (auto i = 1; i <= n; ++i) {
			const float uVal = Get(i, j, u, width) - 0.5f * (Get(i + 1, j, p, width) - Get(i - 1, j, p, width)) / h;
			const float vVal = Get(i, j, v, width) - 0.5f * (Get(i, j + 1, p, width) - Get(i, j - 1, p, width)) / h;
			Set(i, j, u, width, uVal);
			Set(i, j, v, width, vVal);
		}
	}
	SetBoundTangentU(static_cast<int>(n), u);
	SetBoundTangentV(static_cast<int>(n), v);
}

template<typename T>
void FillTexture(ID3D11DeviceContext* context, ID3D11Texture2D* tex, const T* src, size_t srcPitch) {
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	THROW_ON_FAILURE(context->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));

	D3D11_TEXTURE2D_DESC desc;
	tex->GetDesc(&desc);

	for (auto y = size_t(0); y < desc.Height; ++y) {
		auto srcRowBegin = src + y * srcPitch;
		auto destRowBegin = &static_cast<ColorR8G8B8A8*>(mappedResource.pData)[y * mappedResource.RowPitch / sizeof(ColorR8G8B8A8)];
		transform(srcRowBegin, srcRowBegin + desc.Width, destRowBegin, [](const T& val) { 
			return MakeColorR8G8B8A8(Vector3(val)); 
		}); 
	}

	context->Unmap(tex, 0);
}

}

class App : public IApp {
public:
	App(void* hwnd, int width, int height);
	~App();
	void Update() override;


private:
	void DrawTexture(size_t texIndex, const Vector2& offset);
	void UpdateTexture();

	int mWidth;
	int mHeight;

	vector<float> mSources;
	vector<float> mus;
	vector<float> mvs;

	vector<float> mOldHeatVals;
	vector<float> mNewHeatVals;
	vector<Vector3> mColors;

	vector<float> mAfterSources;
	vector<float> mAfterDiffuse;
	vector<float> mAfterAdvect;

	vector<float> mUAfterSources;
	vector<float> mUAfterDiffuse;
	vector<float> mUAfterProject1;
	vector<float> mUAfterAdvect;
	vector<float> mUAfterProject2;

	vector<float> mVAfterSources;
	vector<float> mVAfterDiffuse;
	vector<float> mVAfterProject1;
	vector<float> mVAfterAdvect;
	vector<float> mVAfterProject2;

	mt19937 mGen;

	system_clock::time_point mStartTime;

	void DisplayTurbulence(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height, float time);
	void DisplayCurlNoise(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height, float time);

	void BasicFire(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height);
	void SeedFlames();
	void HeatSpread();

	void StamVelocity(float dt);
	void StamDiffuse(float dt);

	inline void SetOldHeatVal(size_t x, size_t y, float val) { mOldHeatVals[(y + 1) * (mWidth + 2) + (x + 1)] = val; }
	inline float GetOldHeatVal(size_t x, size_t y) { 
		return mOldHeatVals[(y + 1) * (mWidth + 2) + (x + 1)]; 
	}
	inline void SetNewHeatVal(size_t x, size_t y, float val) { mNewHeatVals[(y + 1) * (mWidth + 2) + (x + 1)] = val; }
	inline void SetColor(size_t x, size_t y, const Vector3& color) { mColors[(y + 1) * (mWidth + 2) + (x + 1)] = color; }

	ID3D11DevicePtr mDevice;
	IDXGISwapChainPtr mSwapChain;
	ID3D11DeviceContextPtr mImmediateContext;
	ID3D11Texture2DPtr mBackBuffer;
	ID3D11RenderTargetViewPtr mBackBufferRenderTargetView;
	ID3D11VertexShaderPtr mVertexShader;
	ID3D11PixelShaderPtr mPixelShader;
	ID3D11BufferPtr mVertexBuffer;
	ID3D11InputLayoutPtr mInputLayout;
	ID3D11RasterizerStatePtr mRasterizerState;
	ID3D11DepthStencilStatePtr mDepthStencilState;
	ID3D11Texture2DPtr mTex[4];
	ID3D11ShaderResourceViewPtr mTexView[4];
	ID3D11BufferPtr mVsConstants;
	ID3D11BufferPtr mPsConstants;
	ID3D11SamplerStatePtr mSamplerState;
};

App::App(void* hwnd, int width, int height) :
	mWidth(width),
	mHeight(height),
	mSources((width + 2) * (height + 2), 0.0f),
	mus((width + 2) * (height + 2), 0.0f),
	mvs((width + 2) * (height + 2), -0.2f),
	mOldHeatVals((width + 2) * (height + 2), 0.0f),
	mNewHeatVals((width + 2) * (height + 2), 0.0f),
	mColors((width + 2) * (height + 2)),
	mAfterSources((width + 2) * (height + 2), 0.0f),
	mAfterDiffuse((width + 2) * (height + 2), 0.0f),
	mAfterAdvect((width + 2) * (height + 2), 0.0f),
	mUAfterSources((width + 2) * (height + 2), 0.0f),
	mUAfterDiffuse((width + 2) * (height + 2), 0.0f),
	mUAfterProject1((width + 2) * (height + 2), 0.0f),
	mUAfterAdvect((width + 2) * (height + 2), 0.0f),
	mUAfterProject2((width + 2) * (height + 2), 0.0f),
	mVAfterSources((width + 2) * (height + 2), 0.0f),
	mVAfterDiffuse((width + 2) * (height + 2), 0.0f),
	mVAfterProject1((width + 2) * (height + 2), 0.0f),
	mVAfterAdvect((width + 2) * (height + 2), 0.0f),
	mVAfterProject2((width + 2) * (height + 2), 0.0f),
	mStartTime(system_clock::now())
{
#ifdef _DEBUG
	UINT d3dCreateFlags = D3D11_CREATE_DEVICE_DEBUG;
#else
	UINT d3dCreateFlags = 0;
#endif
	DXGI_SWAP_CHAIN_DESC swapChainDesc;
	ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
	swapChainDesc.BufferDesc.Width = 2 * width;
	swapChainDesc.BufferDesc.Height = 2 * height;
	swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.BufferUsage = DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.BufferCount = 2;
	swapChainDesc.OutputWindow = reinterpret_cast<HWND>(hwnd);
	swapChainDesc.Windowed = TRUE;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	THROW_ON_FAILURE(D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, d3dCreateFlags, NULL, 0, 
		D3D11_SDK_VERSION, &swapChainDesc, &mSwapChain, &mDevice, NULL, &mImmediateContext));
	THROW_ON_FAILURE(mSwapChain->GetBuffer(0, mBackBuffer.GetIID(), reinterpret_cast<void**>(&mBackBuffer)));

	CD3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc(mBackBuffer, D3D11_RTV_DIMENSION_TEXTURE2D);
	THROW_ON_FAILURE(mDevice->CreateRenderTargetView(mBackBuffer, &renderTargetViewDesc, &mBackBufferRenderTargetView));

    CD3D11_VIEWPORT viewPort(0.0f, 0.0f, static_cast<float>(swapChainDesc.BufferDesc.Width), static_cast<float>(swapChainDesc.BufferDesc.Height));
	mImmediateContext->RSSetViewports(1, &viewPort);

	CD3D11_RASTERIZER_DESC rasterizerStateDesc(D3D11_DEFAULT);
	rasterizerStateDesc.CullMode = D3D11_CULL_NONE;
	mDevice->CreateRasterizerState(&rasterizerStateDesc, &mRasterizerState);
	mImmediateContext->RSSetState(mRasterizerState);

	CD3D11_DEPTH_STENCIL_DESC depthStencilStateDesc(D3D11_DEFAULT);
	depthStencilStateDesc.DepthEnable = FALSE;
	mDevice->CreateDepthStencilState(&depthStencilStateDesc, &mDepthStencilState);
	mImmediateContext->OMSetDepthStencilState(mDepthStencilState, 0);

	mVertexShader = LoadVertexShader(mDevice, "vs.cso");
	mInputLayout = CreateInputLayout(mDevice, LoadFile("vs.cso")); 
	mPixelShader = LoadPixelShader(mDevice, "ps.cso");

	const size_t numVertices = 3;
	Vertex vertices[] = {
		{ -1.0f, 1.0f, 0.0f },
		{ 3.0f, 1.0f, 0.0f },
		{ -1.0f, -3.0f, 0.0f }
	};
	mVertexBuffer = CreateVertexBuffer(mDevice, vertices, numVertices);

	CD3D11_TEXTURE2D_DESC texDesc(DXGI_FORMAT_R8G8B8A8_UNORM, width, height);
	texDesc.Usage = D3D11_USAGE_DYNAMIC;
	texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	texDesc.MipLevels = 1;
	for (auto i = 0; i < 4; ++i) {
		THROW_ON_FAILURE(mDevice->CreateTexture2D(&texDesc, NULL, &mTex[i]));
		CD3D11_SHADER_RESOURCE_VIEW_DESC texViewDesc(mTex[i], D3D11_SRV_DIMENSION_TEXTURE2D);
		THROW_ON_FAILURE(mDevice->CreateShaderResourceView(mTex[i], &texViewDesc, &mTexView[i]));
	}

	CD3D11_SAMPLER_DESC samplerDesc(D3D11_DEFAULT);
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	THROW_ON_FAILURE(mDevice->CreateSamplerState(&samplerDesc, &mSamplerState));

	VSConstants vsConstants = { Vector2(0.0f, 0.0f) };
	mVsConstants = CreateConstantBuffer(mDevice, &vsConstants);

	PSConstants psConstants = { Vector2(static_cast<float>(swapChainDesc.BufferDesc.Width), static_cast<float>(swapChainDesc.BufferDesc.Height)),
								Vector2(static_cast<float>(width), static_cast<float>(height)) };
	mPsConstants = CreateConstantBuffer(mDevice, &psConstants);

	for (auto y = mHeight + 1; y > mHeight - 64; --y) {
		//auto lastRowBegin = begin(mSources) + (y) * (mWidth + 2);
		//transform(lastRowBegin, lastRowBegin + mWidth + 2, lastRowBegin, [=](const float val) { return val * (0.99); });
		uniform_int_distribution<int> xDist(64, mWidth - 64);
		for (auto seed = 0; seed < 100; ++seed) {
			Set(xDist(mGen), y, &mSources[0], mWidth + 2, 1.0f);
		}
	}

	for (auto y = 1; y <= mHeight; ++y) {
		for (auto x = 1; x <= mWidth; ++x) {
			Set(x, y, &mus[0], mWidth + 2, Turbulence(x / 128.0f, y / 128.0f, 0.0f) * 0.1f);
			Set(x, y, &mvs[0], mWidth + 2, Turbulence(x / 128.0f, y / 128.0f, 10.0f) * 0.1f - 0.2f);
		}
	}

	mUAfterSources = mus;
	mVAfterSources = mvs;
	mUAfterProject1 = mus;
	mVAfterProject1 = mvs;
	mUAfterProject2 = mus;
	mVAfterProject2 = mvs;
}

App::~App() {
}

void App::SeedFlames() {
	uniform_int_distribution<int> xDist(0, mWidth);
	for (auto seed = 0; seed < 100; ++seed) {
		SetOldHeatVal(xDist(mGen), mHeight, 1.0f);
	}
}

void App::HeatSpread() {
	auto elapsed = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	for (auto y = size_t(1); y < mHeight; ++y) {
		for (auto x = size_t(0); x < mWidth; ++x) {
			const auto coolingFactor = 0.98f + Noise(float(x) / 40.0f, float(y) / 40.0f, elapsed) * 0.02f;
			auto newVal = (GetOldHeatVal(x, y) + GetOldHeatVal(x - 1, y - 1) + GetOldHeatVal(x + 1, y - 1) 
				+ GetOldHeatVal(x + 1, y + 1) + GetOldHeatVal(x - 1, y + 1)) * 0.2f;
			SetNewHeatVal(x, y - 1, newVal * coolingFactor);
		}
	}

	auto lastRowBegin = begin(mNewHeatVals) + (mHeight) * (mWidth + 2);
	transform(lastRowBegin, lastRowBegin + mWidth + 2, lastRowBegin, [=](const float val) { return val * 0.5f; });
}

void App::StamVelocity(const float dt) {
	const float visc = 0.1f;
	AddSource(mWidth, &mUAfterSources[0], &mus[0], dt);
	AddSource(mWidth, &mVAfterSources[0], &mvs[0], dt);
	transform(begin(mUAfterSources), end(mUAfterSources), begin(mVAfterSources), begin(mColors), [](float u, float v) {
		return Vector3(u + 0.5f, v + 0.5f, 0.0f); 
	});
	FillTexture(mImmediateContext, mTex[3], &mColors[mWidth + 2 + 1], mWidth + 2);
	mUAfterDiffuse = mUAfterSources;
	mVAfterDiffuse = mVAfterSources;
	Diffuse(mWidth, &mUAfterDiffuse[0], &mUAfterSources[0], visc, dt, 1);
	Diffuse(mWidth, &mVAfterDiffuse[0], &mVAfterSources[0], visc, dt, 2);
	mUAfterProject1 = mUAfterDiffuse;
	mVAfterProject1 = mVAfterDiffuse;
	Project(mWidth, &mUAfterProject1[0], &mVAfterProject1[0], &mUAfterSources[0], &mVAfterSources[0]);
//	FillTexture(mImmediateContext, mTex[3], &mUAfterProject1[mWidth + 2 + 1], mWidth + 2);
	Advect(mWidth, &mUAfterAdvect[0], &mUAfterProject1[0], &mUAfterProject1[0], &mVAfterProject1[0], dt, 1);
	Advect(mWidth, &mVAfterAdvect[0], &mVAfterProject1[0], &mUAfterProject1[0], &mVAfterProject1[0], dt, 2);
	mUAfterProject2 = mUAfterAdvect;
	mVAfterProject2 = mVAfterAdvect;
//	FillTexture(mImmediateContext, mTex[3], &mVAfterAdvect[mWidth + 2 + 1], mWidth + 2);
	Project(mWidth, &mUAfterProject2[0], &mVAfterProject2[0], &mUAfterProject1[0], &mVAfterProject1[0]);
	mUAfterSources = mUAfterProject2;
	mVAfterSources = mVAfterProject2;
}

void App::StamDiffuse(const float delta) {
//	FillTexture(mImmediateContext, mTex[3], &mSources[mWidth + 2 + 1], mWidth + 2);

	AddSource(mWidth, &mAfterSources[0], &mSources[0], delta);

//	FillTexture(mImmediateContext, mTex[1], &mAfterSources[mWidth + 2 + 1], mWidth + 2);

	mAfterDiffuse = mAfterSources;
	Diffuse(mWidth, &mAfterDiffuse[0], &mAfterSources[0], 1.0f, delta);

//	FillTexture(mImmediateContext, mTex[2], &mAfterDiffuse[mWidth + 2 + 1], mWidth + 2);

	Advect(mWidth, &mAfterAdvect[0], &mAfterDiffuse[0], &mus[0], &mvs[0], delta);

	FillTexture(mImmediateContext, mTex[0], &mAfterAdvect[mWidth + 2 + 1], mWidth + 2);

	mAfterSources = mAfterAdvect;
}

void App::UpdateTexture() {
	static auto startTime = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	auto elapsed = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	auto delta = elapsed - startTime;

	//auto elapsed = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	// BasicFire(data, rowPitch, width, height);
	StamVelocity(delta);
	StamDiffuse(delta);
	//DisplayTurbulence(data, rowPitch, width, height, elapsed);
	//DisplayCurlNoise(data, rowPitch, width, height, elapsed);

	startTime = elapsed;
}

void App::BasicFire(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height) {
	SeedFlames();
	HeatSpread();
	mOldHeatVals.swap(mNewHeatVals);
	
	transform(begin(mNewHeatVals), end(mNewHeatVals), begin(mColors), [](const float val) { 
		auto srgbVal = pow(val, 1.0f / 2.2f); 
		return Vector3(srgbVal, srgbVal, srgbVal); 
	}); 

	for (auto y = size_t(0); y < height; ++y) {
		auto rowBegin = begin(mColors) + (y + 1) * (width + 2) + 1;
		transform(rowBegin, rowBegin + width, &data[y * rowPitch], [](const Vector3& col) { return MakeColorR8G8B8A8(col); }); 
	}
}

void App::DisplayTurbulence(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height, float time) {
	for (auto y = size_t(1); y < mHeight; ++y) {
		for (auto x = size_t(0); x < mWidth; ++x) {
			SetNewHeatVal(x, y, Clamp(Turbulence(x / 128.0f, y / 128.0f, time / 10.0f), -1.0f, 1.0f) * 0.5f + 0.5f);
		}
	}
	

	transform(begin(mNewHeatVals), end(mNewHeatVals), begin(mColors), [](const float val) { 
		auto srgbVal = pow(val, 1.0f / 2.2f); 
		return Vector3(srgbVal, srgbVal, srgbVal); 
	}); 

	for (auto y = size_t(0); y < height; ++y) {
		auto rowBegin = begin(mColors) + y * width;
		transform(rowBegin, rowBegin + width, &data[y * rowPitch], [](const Vector3& col) { return MakeColorR8G8B8A8(col); }); 
	}
}

void App::DisplayCurlNoise(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height, float time) {
	for (auto y = size_t(1); y < mHeight; ++y) {
		for (auto x = size_t(0); x < mWidth; ++x) {
			const auto scaledTime = time / 10.0f;
			SetColor(x, y, Clamp(CurlNoise(x / 128.0f, y / 128.0f, scaledTime), -1.0f, 1.0f) * 0.5f + 0.5f);
		}
	}

	for (auto y = size_t(0); y < height; ++y) {
		auto rowBegin = begin(mColors) + y * width;
		transform(rowBegin, rowBegin + width, &data[y * rowPitch], [](const Vector3& col) { return MakeColorR8G8B8A8(col); }); 
	}
}

void App::DrawTexture(size_t texIndex, const Vector2& offset)
{
	VSConstants vsConstants = { offset };
	mImmediateContext->UpdateSubresource(mVsConstants, 0, NULL, &vsConstants, 0, 0);
	ID3D11ShaderResourceView* texView = mTexView[texIndex];
	mImmediateContext->PSSetShaderResources(0, 1, &texView);
	mImmediateContext->Draw(3, 0);
}

void App::Update() {
	const FLOAT color[] = { 0.1f, 0.1f, 1.0f, 1.0f };
	mImmediateContext->ClearRenderTargetView(mBackBufferRenderTargetView, color);

	ID3D11RenderTargetView* renderTargetView = mBackBufferRenderTargetView;
	mImmediateContext->OMSetRenderTargets(1, &renderTargetView, NULL);

	UpdateTexture();

	mImmediateContext->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetInputLayout(mInputLayout);
	ID3D11Buffer* vertexBuffers[] = { mVertexBuffer };
	UINT strides[] = { sizeof(Vertex) };
	UINT offsets[] = { 0 };
	mImmediateContext->IASetVertexBuffers(0, 1, vertexBuffers, strides, offsets);
	ID3D11Buffer* vsConstants = mVsConstants;
	mImmediateContext->VSSetConstantBuffers(0, 1, &vsConstants);
	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);
	ID3D11SamplerState* samplerState = mSamplerState;
	mImmediateContext->PSSetSamplers(0, 1, &samplerState);
	ID3D11Buffer* psConstants = mPsConstants;
	mImmediateContext->PSSetConstantBuffers(0, 1, &psConstants);
	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	DrawTexture(0, Vector2(0.0f, 0.0f));
	DrawTexture(1, Vector2(1.0f, 0.0f));
	DrawTexture(2, Vector2(0.0f, -1.0f));
	DrawTexture(3, Vector2(1.0f, -1.0f));

	mSwapChain->Present(1, 0);
}

unique_ptr<IApp> CreateApp(void* hwnd, int width, int height) {
	return unique_ptr<IApp>(new App(hwnd, width, height));
}
