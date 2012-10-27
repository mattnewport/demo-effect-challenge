#include "app.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>
#include <numeric>

#include <comdef.h>
#include <d3d11.h>

#include <emmintrin.h>

extern "C" void AddSourceAsm(size_t n, float* x, const float* x0, float dt);
extern "C" void RelaxAsm(size_t n, float* x, const float* x0, const float a, const float b, const int bounds);

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

__m128 ClampSSE(__m128 x, __m128 a, __m128 b) {
    return _mm_max_ps(_mm_min_ps(x, b), a);
}

struct ColorR8G8B8A8 {
	union {
		uint32_t rgba;
		struct { uint8_t r, g, b, a; };
	};
};

inline ColorR8G8B8A8 MakeColorR8G8B8A8(float r, float g, float b, float a) {
    const __m128 mmZero = _mm_setzero_ps();
    const __m128 mmOne = _mm_set_ps1(1.0f);
    const __m128 mm255 = _mm_set_ps1(255.0f);
    __m128 mmCol = ClampSSE(_mm_set_ps(r, g, b, a), mmZero, mmOne);
    __m128 mmGamma = _mm_sqrt_ps(mmCol);
    __m128 mmScaled = _mm_mul_ps(mmGamma, mm255);
    __m128i mmInt = _mm_cvtps_epi32(mmScaled);
	ColorR8G8B8A8 col;
	col.r = static_cast<uint8_t>(mmInt.m128i_i32[3]);
	col.g = static_cast<uint8_t>(mmInt.m128i_i32[2]);
	col.b = static_cast<uint8_t>(mmInt.m128i_i32[1]);
	col.a = static_cast<uint8_t>(mmInt.m128i_i32[0]);
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

float BilerpSSEss(float s, float t, float v00, float v10, float v01, float v11) {
    __m128 mms = _mm_load1_ps(&s);
    __m128 mmt = _mm_load1_ps(&t);
    __m128 mmv00 = _mm_load1_ps(&v00);
    __m128 mmv10 = _mm_load1_ps(&v10);
    __m128 mmv01 = _mm_load1_ps(&v01);
    __m128 mmv11 = _mm_load1_ps(&v11);

    __m128 t0 = _mm_sub_ps(mmv10, mmv00);
    t0 = _mm_mul_ps(mms, t0);
    t0 = _mm_add_ps(mmv00, t0);

    __m128 t1 = _mm_sub_ps(mmv11, mmv01);
    t1 = _mm_mul_ps(mms, t1);
    t1 = _mm_add_ps(mmv01, t1);

    __m128 t2 = _mm_sub_ps(t1, t0);
    t2 = _mm_mul_ps(mmt, t2);
    t2 = _mm_add_ps(t0, t2);

    return t2.m128_f32[0];
}

inline __m128 BilerpSSEps(__m128 mms, __m128 mmt, __m128 mmv00, __m128 mmv10, __m128 mmv01, __m128 mmv11) {
	__m128 t0 = _mm_sub_ps(mmv10, mmv00);
	t0 = _mm_mul_ps(mms, t0);
	t0 = _mm_add_ps(mmv00, t0);

	__m128 t1 = _mm_sub_ps(mmv11, mmv01);
	t1 = _mm_mul_ps(mms, t1);
	t1 = _mm_add_ps(mmv01, t1);

	__m128 t2 = _mm_sub_ps(t1, t0);
	t2 = _mm_mul_ps(mmt, t2);
	t2 = _mm_add_ps(t0, t2);

	return t2;
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

extern "C" void SetBoundSSE(int n, float* x, int bounds) {
    const int width = n + 2;
    const int nPlus1 = n + 1;

    const float one = 1.0f;
    const float minusOne = -1.0f;
    const float bx = (bounds == 1) ? minusOne : one;
    __m128 by = (bounds == 2) ? _mm_load1_ps(&minusOne) : _mm_load1_ps(&one);
    
    const int bottomRowOffset = (n + 1) * width;
    assert((n % 4) == 0);
    for (int i = 1; i <= n; i += 4) {
        __m128 t0 = _mm_loadu_ps(x + width + i);
        t0 = _mm_mul_ps(t0, by);
        _mm_storeu_ps(x + i, t0);
        t0 = _mm_loadu_ps(x + bottomRowOffset - width + i);
        t0 = _mm_mul_ps(t0, by);
        _mm_storeu_ps(x + bottomRowOffset + i, t0);
    }

    for (int i = 1; i <= n; ++i) {
        x[i * width] = bx * x[i * width + 1];
        x[i * width + nPlus1] = bx * x[i * width + n];
    }

    x[0] = 0.5f * (x[width] + x[1]);
    x[width * nPlus1] = 0.5f * (x[width * n] + x[width * nPlus1 + 1]);
    x[nPlus1] = 0.5f * (x[width + nPlus1] + x[n]);
    x[width * nPlus1 + nPlus1] = 0.5f * (x[width * n + nPlus1] + x[width * nPlus1 + n]);
}

void AddSource(size_t n, float* x, const float* x0, float dt) {
	const auto size = (n + 2) * (n + 2);
	for (auto i = 0; i < size; ++i) {
		x[i] += x0[i] * dt;
	}
}

__declspec(noinline) void AddSourceSSE(size_t n, float* x, const float* x0, float dt) {
    const size_t nPlus2 = n + 2;
    const size_t size = nPlus2 * nPlus2;
    assert(intptr_t(x) % 16 == 0);
    assert((size % 4) == 0);
    __m128 mmdt = _mm_load1_ps(&dt);
    for (size_t i = 0; i < size; i += 4) {
        __m128 t0 = _mm_load_ps(x0);
        t0 = _mm_mul_ps(t0, mmdt);
        __m128 t1 = _mm_load_ps(x);
        t0 = _mm_add_ps(t0, t1);
        _mm_store_ps(x, t0);
        x += 4;
        x0 += 4;
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

__declspec(noinline) void RelaxSSE(size_t n, float* x, const float* x0, const float a, const float b, const int bounds = 0) {
    const size_t width = n + 2;
    const float bPlus4aRcp = 1.0f / (b + 4.0f * a);
    const __m128 mmA = _mm_load1_ps(&a);
    const __m128 mmBPlus4aRcp = _mm_load1_ps(&bPlus4aRcp);
    for (size_t k = 0; k < 20; ++k) {
        for (size_t j = 1; j <= n; ++j) {
            for (size_t i = 1; i <= n; i += 4) {
                const size_t offset = j * width + i;
                __m128 t0 = _mm_loadu_ps(x + offset - 1);
                __m128 t1 = _mm_loadu_ps(x + offset + 1);
                t0 = _mm_add_ps(t0, t1);
                t1 = _mm_loadu_ps(x + offset - width);
				t0 = _mm_add_ps(t0, t1);
                t1 = _mm_loadu_ps(x + offset + width);
                t0 = _mm_add_ps(t0, t1);
                t0 = _mm_mul_ps(t0, mmA);
				t1 = _mm_loadu_ps(x0 + offset);
                t0 = _mm_add_ps(t0, t1);
                t0 = _mm_mul_ps(t0, mmBPlus4aRcp);
                _mm_storeu_ps(x + offset, t0);
            }
        }
        SetBoundSSE(static_cast<int>(n), x, bounds);
    }
}

void DiffuseSSE(size_t n, float* x, const float* x0, float diff, float dt, const int bounds = 0) {
	const float a = dt * diff * n * n;
	RelaxAsm(n, x, x0, a, 1.0f, bounds);
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
	SetBoundSSE(static_cast<int>(n), d, bounds);
}

void AdvectSSE(const size_t n, float* d, const float* d0, const float* u, const float* v, const float dt, const int bounds = 0) {
    const float dt0 = dt * n;
    const int width = static_cast<int>(n) + 2;
    const float low = 0.5f;
    const float high = n + 0.5f;
    const __m128 mmlow = _mm_load1_ps(&low);
    const __m128 mmhigh = _mm_load1_ps(&high);
    const __m128 mmdt0 = _mm_load1_ps(&dt0);
    
    for (int j = 1; j <= n; ++j) {
        for (int i = 1; i <= n; i += 4) {
            const int offset = j * width + i;

			__m128i mmi = _mm_set_epi32(i + 3, i + 2, i + 1, i);
			__m128i mmj = _mm_set_epi32(j, j, j, j);
			__m128 mmfi = _mm_cvtepi32_ps(mmi);
			__m128 mmfj = _mm_cvtepi32_ps(mmj);
            __m128 mmx = ClampSSE(_mm_sub_ps(mmfi, _mm_mul_ps(mmdt0, _mm_loadu_ps(u + offset))), mmlow, mmhigh);
            __m128 mmy = ClampSSE(_mm_sub_ps(mmfj, _mm_mul_ps(mmdt0, _mm_loadu_ps(v + offset))), mmlow, mmhigh);
			__m128i mmi0 = _mm_cvttps_epi32(mmx);
			__m128i mmj0 = _mm_cvttps_epi32(mmy);
			__m128 mmtl, mmtr, mmbl, mmbr;
			for (int elem = 0; elem < 4; ++elem) {
				const int elemOffset = mmj0.m128i_i32[elem] * width + mmi0.m128i_i32[elem];
				mmtl.m128_f32[elem] = d0[elemOffset];
				mmtr.m128_f32[elem] = d0[elemOffset + 1];
				mmbl.m128_f32[elem] = d0[elemOffset + width];
				mmbr.m128_f32[elem] = d0[elemOffset + width + 1];
			}
			_mm_storeu_ps(d + offset, BilerpSSEps(_mm_sub_ps(mmx, _mm_cvtepi32_ps(mmi0)), _mm_sub_ps(mmy, _mm_cvtepi32_ps(mmj0)), mmtl, mmtr, mmbl, mmbr));
        }
    }
    SetBoundSSE(static_cast<int>(n), d, bounds);
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

	SetBoundSSE(static_cast<int>(n), div, 0);
	SetBoundSSE(static_cast<int>(n), p, 0);

	Relax(n, p, div, 1.0f, 0.0f);

	for (auto j = 1; j <= n; ++j) {
		for (auto i = 1; i <= n; ++i) {
			const float uVal = Get(i, j, u, width) - 0.5f * (Get(i + 1, j, p, width) - Get(i - 1, j, p, width)) / h;
			const float vVal = Get(i, j, v, width) - 0.5f * (Get(i, j + 1, p, width) - Get(i, j - 1, p, width)) / h;
			Set(i, j, u, width, uVal);
			Set(i, j, v, width, vVal);
		}
	}
	SetBoundSSE(static_cast<int>(n), u, 1);
    SetBoundSSE(static_cast<int>(n), v, 2);
}

void ProjectSSE(size_t n, float* const u, float* const v, float* const p, float* const div)
{
    const float h = 1.0f / n;
    const float minusHalfH = -0.5f * h;
    const float halfOverH = 0.5f / h;
    //const __m128 mmH = _mm_load1_ps(&h);
    const __m128 mmMinusHalfH = _mm_load1_ps(&minusHalfH);
    const __m128 mmHalfOverH = _mm_load1_ps(&halfOverH);
    const __m128 mmZero = _mm_setzero_ps();
    const int width = static_cast<int>(n) + 2;

    for (int j = 1; j <= n; ++j) {
        for (int i = 1; i <= n; i += 4) {
            const int offset = j * width + i;
            __m128 ul = _mm_loadu_ps(u + offset - 1);
            __m128 ur = _mm_loadu_ps(u + offset + 1);
            __m128 vt = _mm_loadu_ps(v + offset - width);
            __m128 vb = _mm_loadu_ps(v + offset + width);
            __m128 t0 = _mm_sub_ps(ur, ul);
            __m128 t1 = _mm_sub_ps(vb, vt);
            t0 = _mm_add_ps(t0, t1);
            t0 = _mm_mul_ps(mmMinusHalfH, t0);
            _mm_storeu_ps(div + offset, t0);
            _mm_storeu_ps(p + offset, mmZero);
        }
    }

    SetBoundSSE(static_cast<int>(n), div, 0);
    SetBoundSSE(static_cast<int>(n), p, 0);

    RelaxAsm(n, p, div, 1.0f, 0.0f, 0);

    for (int j = 1; j <= n; ++j) {
        for (int i = 1; i <= n; i += 4) {
            const int offset = j * width + i;

            __m128 pl = _mm_loadu_ps(p + offset - 1);
            __m128 pr = _mm_loadu_ps(p + offset + 1);
            __m128 t0 = _mm_sub_ps(pr, pl);
            t0 = _mm_mul_ps(mmHalfOverH, t0);
            __m128 mmu = _mm_loadu_ps(u + offset);
            t0 = _mm_sub_ps(mmu, t0);
            _mm_storeu_ps(u + offset, t0);

            __m128 pt = _mm_loadu_ps(p + offset - width);
            __m128 pb = _mm_loadu_ps(p + offset + width);
            t0 = _mm_sub_ps(pb, pt);
            t0 = _mm_mul_ps(mmHalfOverH, t0);
            __m128 mmv = _mm_loadu_ps(v + offset);
            t0 = _mm_sub_ps(mmv, t0);
            _mm_storeu_ps(v + offset, t0);
        }
    }
    SetBoundSSE(static_cast<int>(n), u, 1);
    SetBoundSSE(static_cast<int>(n), v, 2);
}

void DensityStep(size_t n, float* d, float* d0, const float* u, const float* v, float diff, float dt)
{
	AddSourceAsm(n, d, d0, dt);

	swap(d, d0);
	DiffuseSSE(n, d, d0, diff, dt);

	swap(d, d0);
	AdvectSSE(n, d, d0, u, v, dt);
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

float BlackBodyIntensity(float lambda, float t) {
    const auto h = 6.626e-34f; // Planck's constant
    const auto c = 299792458.0f; // speed of light
    const auto k = 1.380649e-23f; // Boltzmann constant
    const auto num = 2.0f * 3.14159f * h * c * c;
    const auto denom = pow(lambda, 5.0f) * (exp((h * c) / (lambda * k * t)) - 1.0f);
    return num / denom;
}

Vector3 BlackBodyColor(float t) {
    const float r = BlackBodyIntensity(700e-9f, t);
    const float g = BlackBodyIntensity(546e-9f, t);
    const float b = BlackBodyIntensity(436e-9f, t);
    return Vector3(r, g, b);
}

float ExponentialMapping(float x) {
    const float lAvg = 1000.0f;
    return 1.0f - exp(-x / lAvg);
}

Vector3 ExponentialMapping(const Vector3& v) {
    return Vector3(ExponentialMapping(v.x), ExponentialMapping(v.y), ExponentialMapping(v.z));
}

struct Params {
    Params() {
        alpha = 0.05f; // fuel / exhaust dissipation rate
        heatAlpha = 0.05f; // heat disspation rate
        g = 9.8f * 0.001f; // force due to gravity
        b = -0.000033f; // buoyancy force
        ke = 0.00005f;//0.2f * 0.01f; // expansion constant
        tAmbient = 0.0f; // ambient temp
        r = -log(1.0f - 0.3f); // Burn rate
        bMix = 1.0f;
        tThreshold = 300.0f; // combustion threshold temp
        t0 = 50000.0f; // temperature rise due to combustion
    }
    float alpha;
    float heatAlpha;
    float g;
    float b;
    float ke;
    float tAmbient;
    float r;
    float bMix;
    float tThreshold;
    float t0;
};

}

class App : public IApp {
public:
	App(void* hwnd, int width, int height);
	~App();
	void Update(bool mouseButtons[2], int mouseX, int mouseY) override;


private:
	void DrawTexture(size_t texIndex, const Vector2& offset);
	void UpdateTexture(bool mouseButtons[2], int mouseX, int mouseY);

	int mWidth;
	int mHeight;

	vector<float> mHeatSources;
	vector<float> mFuelSources;
	vector<float> mExhaustSources;
	vector<float> mUTurb;
	vector<float> mVTurb;
	vector<float> mus[2];
	vector<float> mvs[2];
	vector<float> mHeat[2];
	vector<float> mFuel[2];
	vector<float> mExhaust[2];

	vector<float> mOldHeatVals;
	vector<float> mNewHeatVals;
	vector<Vector3> mColors;

	vector<float> mAfterSources;
	vector<float> mAfterDiffuse;
	vector<float> mAfterAdvect;

    vector<float> mFrameTimes;

	mt19937 mGen;

	system_clock::time_point mStartTime;

	vector<char> mDataFile;

	void DisplayTurbulence(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height, float time);
	void DisplayCurlNoise(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height, float time);

	void BasicFire(ColorR8G8B8A8* const data, size_t rowPitch, size_t width, size_t height);
	void SeedFlames();
	void HeatSpread();

	void AddForces(const Params& p);
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

	const float diff;
    const float heatDiff;
	const float visc;
	const float force;
	const float source;

	// No copy or assignment
	App(const App&);
	App operator=(const App&);

};

App::App(void* hwnd, int width, int height) :
	mWidth(width),
	mHeight(height),
	mHeatSources((width + 2) * (height + 2), 0.0f),
	mFuelSources((width + 2) * (height + 2), 0.0f),
	mExhaustSources((width + 2) * (height + 2), 0.0f),
	mOldHeatVals((width + 2) * (height + 2), 0.0f),
	mNewHeatVals((width + 2) * (height + 2), 0.0f),
	mColors((width + 2) * (height + 2)),
	mAfterSources((width + 2) * (height + 2), 0.0f),
	mAfterDiffuse((width + 2) * (height + 2), 0.0f),
	mAfterAdvect((width + 2) * (height + 2), 0.0f),
	mStartTime(system_clock::now()),
	diff(0.0001f),
    heatDiff(0.0005f),
	visc(0.0f),
	force(0.2f),
	source(1000.0f)
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
	mDataFile = LoadFile("c:\\Users\\mnewport\\Dropbox\\Projects\\demo-effect-challenge\\data-gen\\bin\\Release\\data.bin");

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

/*
	for (auto y = mHeight + 1; y > mHeight - 64; --y) {
		//auto lastRowBegin = begin(mSources) + (y) * (mWidth + 2);
		//transform(lastRowBegin, lastRowBegin + mWidth + 2, lastRowBegin, [=](const float val) { return val * (0.99); });
		uniform_int_distribution<int> xDist(64, mWidth - 64);
		for (auto seed = 0; seed < 10; ++seed) {
			Set(xDist(mGen), y, &mHeatSources[0], mWidth + 2, source);
		}
	}
*/

	const int dim = (width + 2) * (height + 2);
	mHeat[0].resize(dim);
	fill(begin(mHeat[0]), end(mHeat[0]), 0.0f);
	mHeat[1].resize(dim);
	fill(begin(mHeat[1]), end(mHeat[1]), 0.0f);

	mFuel[0].resize(dim);
	fill(begin(mFuel[0]), end(mFuel[0]), 0.0f);
	mFuel[1].resize(dim);
	fill(begin(mFuel[1]), end(mFuel[1]), 0.0f);

	mExhaust[0].resize(dim);
	fill(begin(mExhaust[0]), end(mExhaust[0]), 0.0f);
	mExhaust[1].resize(dim);
	fill(begin(mExhaust[1]), end(mExhaust[1]), 0.0f);

	//float* fuelDataFloat = reinterpret_cast<float*>(&mDataFile[0]);
	mUTurb.resize(dim);
	mVTurb.resize(dim);
	fill(begin(mUTurb), end(mUTurb), 0.0f);
	fill(begin(mVTurb), end(mVTurb), 0.0f);
	for (auto y = 1; y <= mHeight; ++y) {
		for (auto x = 1; x <= mWidth; ++x) {
			Set(x, y, &mUTurb[0], mWidth + 2, Turbulence(x / 128.0f, y / 128.0f, 0.0f) * force);
			Set(x, y, &mVTurb[0], mWidth + 2, Turbulence(x / 128.0f, y / 128.0f, 10.0f) * force);
			//Set(x, y, &mHeatSources[0], mWidth + 2, Turbulence(x / 128.0f, y / 128.0f, 20.0f) * 100.0f + 300.0f);
			//Set(x, y, &mFuelSources[0], mWidth + 2, fuelDataFloat[y * mWidth + x]);
		}
	}

	mus[0] = mUTurb;
	mvs[0] = mVTurb;
	mus[1].resize(dim);
	mvs[1].resize(dim);
	fill(begin(mus[1]), end(mus[1]), 0.0f);
	fill(begin(mvs[1]), end(mvs[1]), 0.0f);
	mus[1] = mus[0];
	mvs[1] = mvs[0];

	mHeat[1] = mHeat[0] = mHeatSources;
	//mFuel[1] = mFuel[0] = mFuelSources;
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

void App::AddForces(const Params& p) {
	const int n = mWidth;

// vorticity confinement
#if 0
	const float h = 1.0f / n;
	const float eps = 1000.0f * dt;
	const float epsh = eps * h;

	for (auto j = 1; j <= n; j += 2) {
		for (auto i = 1; i <= n; i += 2) {
			const size_t width = n + 2;
			
			const float x0y0U = Get(i, j, &mus[1][0], width);
			const float x0y0V = Get(i, j, &mvs[1][0], width);
			const float x1y0U = Get(i + 1, j, &mus[1][0], width);
			const float x1y0V = Get(i + 1, j, &mvs[1][0], width);
			const float x1y1U = Get(i + 1, j + 1, &mus[1][0], width);
			const float x1y1V = Get(i + 1, j + 1, &mvs[1][0], width);
			const float x0y1U = Get(i, j + 1, &mus[1][0], width);
			const float x0y1V = Get(i, j + 1, &mvs[1][0], width);

			const float dvdx0 = x1y0V - x0y0V;
			const float dvdx1 = x1y1V - x0y1V;
			const float dudy0 = x0y1U - x0y0U;
			const float dudy1 = x1y1U - x1y0U;

			const float w00 = dvdx0 - dudy0;
			const float w10 = -dvdx0 - dudy1;
			const float w11 = -dvdx1 + dudy1;
			const float w01 = dvdx1 + dudy0;

			const float dwx0 = w10 - w00;
			const float dwx1 = w11 - w01;
			const float dwy0 = w01 - w00;
			const float dwy1 = w11 - w10;

			const float nx00 = dwx0 / sqrt(dwx0 * dwx0 + dwy0 * dwy0);
			const float ny00 = dwy0 / sqrt(dwx0 * dwx0 + dwy0 * dwy0);
			const float nx10 = -dwx0 / sqrt(dwx0 * dwx0 + dwy1 * dwy1);
			const float ny10 = dwy1 / sqrt(dwx0 * dwx0 + dwy1 * dwy1);
			const float nx11 = -dwx1 / sqrt(dwx1 * dwx1 + dwy1 * dwy1);
			const float ny11 = -dwy1 / sqrt(dwx1 * dwx1 + dwy1 * dwy1);
			const float nx01 = dwx1 / sqrt(dwx1 * dwx1 + dwy0 * dwy0);
			const float ny01 = -dwy0 / sqrt(dwx1 * dwx1 + dwy0 * dwy0);

			Set(i, j, &mUTurb[0], width, Get(i, j, &mUTurb[0], width) + w00 * ny00 * epsh);
			Set(i, j, &mVTurb[0], width, Get(i, j, &mVTurb[0], width) + w00 * nx00 * epsh);
			Set(i + 1, j, &mUTurb[0], width, Get(i, j, &mUTurb[0], width) + w10 * ny10 * epsh);
			Set(i + 1, j, &mVTurb[0], width, Get(i, j, &mVTurb[0], width) + w10 * nx10 * epsh);
			Set(i + 1, j + 1, &mUTurb[0], width, Get(i, j, &mUTurb[0], width) + w11 * ny11 * epsh);
			Set(i + 1, j + 1, &mVTurb[0], width, Get(i, j, &mVTurb[0], width) + w11 * nx11 * epsh);
			Set(i, j + 1, &mUTurb[0], width, Get(i, j, &mUTurb[0], width) + w01 * ny01 * epsh);
			Set(i, j + 1, &mVTurb[0], width, Get(i, j, &mVTurb[0], width) + w01 * nx01 * epsh);
		}

	}

	transform(begin(mUTurb), end(mUTurb), begin(mVTurb), begin(mColors), [](float u, float v) {
		return Vector3(u + 0.5f, v + 0.5f, 0.0f); 
	});
	FillTexture(mImmediateContext, mTex[2], &mColors[mWidth + 2 + 1], mWidth + 2);
#endif

	for (auto j = 1; j <= n; ++j) {
		for (auto i = 1; i <= n; ++i) {
			const size_t width = n + 2;
			const float t = Get(i, j, &mHeat[1][0], width);

			const auto fGravity = p.g * (Get(i, j, &mFuel[1][0], width) + Get(i, j, &mExhaust[1][0], width));
			const auto fBuoyancy = p.b * (Get(i, j, &mHeat[1][0], width) - p.tAmbient);
            const auto fExpansionU = t > p.tThreshold ? p.ke * (float(i) - float(width / 2)) : 0.0f;
			const auto fExpansionV = t > p.tThreshold ? p.ke * (float(j) - float(width / 2)) : 0.0f;
			Set(i, j, &mUTurb[0], width, Get(i, j, &mUTurb[0], width) + fExpansionU);
			Set(i, j, &mVTurb[0], width, Get(i, j, &mVTurb[0], width) + fGravity + fBuoyancy + fExpansionV);
		}
	}

	for (auto j = 1; j <= n; ++j) {
		for (auto i = 1; i <= n; ++i) {
			const size_t width = n + 2;
			const float g = Get(i, j, &mFuel[1][0], width);
			const float t = Get(i, j, &mHeat[1][0], width);
			const float e = Get(i, j, &mExhaust[1][0], width);

			const float c = p.r * p.bMix * g;
			float cg = 0.0f;
			float ca = 0.0f;
			float ct = 0.0f;
			if (t > p.tThreshold) {
				cg = (-c / p.bMix);
				ca = c * (1.0f + 1.0f / p.bMix);
				ct = p.t0 * c;
			}
// 			const float cg = t > tThreshold ? (-c / bMix) : 0.0f;
// 			const float ca = t > tThreshold ? c * (1.0f + 1.0f / bMix) : 0.0f;
// 			const float ct = t > tThreshold ? t0 * c : 0.0f;

			Set(i, j, &mFuelSources[0], width, Get(i, j, &mFuelSources[0], width) + cg - p.alpha * g);
			Set(i, j, &mExhaustSources[0], width, Get(i, j, &mExhaustSources[0], width) + ca - p.alpha * e);
			Set(i, j, &mHeatSources[0], width, Get(i, j, &mHeatSources[0], width) + ct - p.heatAlpha * t);
		}
	}
}

void App::StamVelocity(const float dt) {
	mus[0] = mUTurb;
	mvs[0] = mVTurb;
	AddSourceAsm(mWidth, &mus[1][0], &mus[0][0], dt);
	AddSourceAsm(mWidth, &mvs[1][0], &mvs[0][0], dt);

	mus[0].swap(mus[1]);
	DiffuseSSE(mWidth, &mus[1][0], &mus[0][0], visc, dt, 1);
	mvs[0].swap(mvs[1]);
	DiffuseSSE(mWidth, &mvs[1][0], &mvs[0][0], visc, dt, 2);
	ProjectSSE(mWidth, &mus[1][0], &mvs[1][0], &mus[0][0], &mvs[0][0]);

	mus[0].swap(mus[1]);
	mvs[0].swap(mvs[1]);
	AdvectSSE(mWidth, &mus[1][0], &mus[0][0], &mus[0][0], &mvs[0][0], dt, 1);
	AdvectSSE(mWidth, &mvs[1][0], &mvs[0][0], &mus[0][0], &mvs[0][0], dt, 2);
	ProjectSSE(mWidth, &mus[1][0], &mvs[1][0], &mus[0][0], &mvs[0][0]);

	transform(begin(mus[1]), end(mus[1]), begin(mvs[1]), begin(mColors), [](float u, float v) {
		return Vector3(u + 0.5f, v + 0.5f, 0.0f); 
	});
	FillTexture(mImmediateContext, mTex[3], &mColors[mWidth + 2 + 1], mWidth + 2);
}

void App::StamDiffuse(const float delta) {
	mHeat[0] = mHeatSources;
	mFuel[0] = mFuelSources;
	mExhaust[0] = mExhaustSources;
	DensityStep(mWidth, &mHeat[1][0], &mHeat[0][0], &mus[1][0], &mvs[1][0], heatDiff, delta);
	DensityStep(mWidth, &mFuel[1][0], &mFuel[0][0], &mus[1][0], &mvs[1][0], diff, delta);
	DensityStep(mWidth, &mExhaust[1][0], &mExhaust[0][0], &mus[1][0], &mvs[1][0], diff, delta);

    double sum = 0.0f;
    float minTemp = 1e30f;
    float maxTemp = -1e30f;
    for_each(begin(mHeat[1]), end(mHeat[1]), [&](float heatVal) {
        sum += heatVal;
        minTemp = min(heatVal, minTemp);
        maxTemp = max(heatVal, maxTemp);
    });
    double averageTemp = sum / mHeat[1].size();
    cout << "Min, Max, Average Temp:" << minTemp << ", " << maxTemp << ", " << averageTemp << endl;
    float heatOffset = minTemp;
    float heatScale = 1.0f / (maxTemp - minTemp);

    const ColorR8G8B8A8* colorTable = reinterpret_cast<const ColorR8G8B8A8*>(&mDataFile[sizeof(Params) + mWidth * mHeight * sizeof(float)]);
    transform(begin(mHeat[1]), end(mHeat[1]), begin(mExhaust[1]), begin(mColors), [=](float t, float e) {
        //return Vector3((t - heatOffset) * heatScale * e);//ExponentialMapping(BlackBodyColor((t + 300.0f) * e * 2.0f)); 
        int colorTableIdx = Clamp((int)((t - heatOffset) * heatScale * 511.0f), 0, 511);
        auto color = colorTable[colorTableIdx];
        return Vector3((float)color.r / 255.0f, (float)color.g / 255.0f, (float)color.b / 255.0f) * e;
    });
	FillTexture(mImmediateContext, mTex[0], &mColors[mWidth + 2 + 1], mWidth + 2);
	FillTexture(mImmediateContext, mTex[1], &mExhaust[1][mWidth + 2 + 1], mWidth + 2);
/*
	transform(begin(mHeat[1]), end(mHeat[1]), begin(mColors), [](float t) {
		return Vector3(t * 0.002f); 
	});
*/
	FillTexture(mImmediateContext, mTex[2], &mFuel[1][mWidth + 2 + 1], mWidth + 2);
}

void App::UpdateTexture(bool mouseButtons[2], int mouseX, int mouseY) {
	static auto startTime = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	auto elapsed = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	auto delta = elapsed - startTime;
    mFrameTimes.insert(begin(mFrameTimes), delta * 1000.0f);
    if (mFrameTimes.size() > 100)
        mFrameTimes.pop_back();
    auto frameTimesSum = accumulate(begin(mFrameTimes), end(mFrameTimes), 0.0f);
    auto frameTimesRunningAverage = frameTimesSum / mFrameTimes.size();
    cout << "Average frame time: " << frameTimesRunningAverage << "ms" << endl;

	if (mouseX > 0 && mouseX < mWidth && mouseY > 0 && mouseY < mHeight) {
		static int oldMouseX = mouseX;
		static int oldMouseY = mouseY;

		if (mouseButtons[0] && mouseButtons[1]) {
			for (int y = Clamp(mouseY - 5, 1, mHeight + 1); y < Clamp(mouseY + 5, 1, mHeight + 1); ++y) {
				for (int x = Clamp(mouseX - 5, 1, mWidth + 1); x < Clamp(mouseX + 5, 1, mWidth + 1); ++x) {
					Set(x, y, &mHeatSources[0], mWidth + 2, source);
				}
			}
		}
		else if (mouseButtons[0]) {
			Set(mouseX, mouseY, &mUTurb[0], mWidth + 2, force * 10.0f * (mouseX - oldMouseX));
			Set(mouseX, mouseY, &mVTurb[0], mWidth + 2, force * 10.0f * (mouseY - oldMouseY));
		}
		else if (mouseButtons[1]) {
			for (int y = Clamp(mouseY - 10, 1, mHeight + 1); y < Clamp(mouseY + 10, 1, mHeight + 1); ++y) {
				for (int x = Clamp(mouseX - 10, 1, mWidth + 1); x < Clamp(mouseX + 10, 1, mWidth + 1); ++x) {
					Set(x, y, &mFuelSources[0], mWidth + 2, 2.0f);
				}
			}
		}

		if (mouseButtons[0] || mouseButtons[1]) {
			oldMouseX = mouseX;
			oldMouseY = mouseY;
		}
	}

	//auto elapsed = static_cast<float>(duration_cast<milliseconds>(system_clock::now() - mStartTime).count()) / 1000.0f;
	// BasicFire(data, rowPitch, width, height);
    const Params* p = reinterpret_cast<const Params*>(&mDataFile[0]);
	AddForces(*p);
	StamVelocity(delta);
	StamDiffuse(delta);
	//DisplayTurbulence(data, rowPitch, width, height, elapsed);
	//DisplayCurlNoise(data, rowPitch, width, height, elapsed);

	fill(begin(mUTurb), end(mUTurb), 0.0f);
	fill(begin(mVTurb), end(mVTurb), 0.0f);
	fill(begin(mHeatSources), end(mHeatSources), 0.0f);
	fill(begin(mFuelSources), end(mFuelSources), 0.0f);
	fill(begin(mExhaustSources), end(mExhaustSources), 0.0f);

    const float modulate = (sin(elapsed) + 2.0f);
	float* fuelDataFloat = reinterpret_cast<float*>(&mDataFile[sizeof(Params)]);
	for (auto y = 1; y <= mHeight; ++y) {
		for (auto x = 1; x <= mWidth; ++x) {
			Set(x, y, &mFuelSources[0], mWidth + 2, fuelDataFloat[(y - 1) * mWidth + (x - 1)] * 0.1f * modulate);
		}
	}

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

void App::Update(bool mouseButtons[2], int mouseX, int mouseY) {
	const FLOAT color[] = { 0.1f, 0.1f, 1.0f, 1.0f };
	mImmediateContext->ClearRenderTargetView(mBackBufferRenderTargetView, color);

	ID3D11RenderTargetView* renderTargetView = mBackBufferRenderTargetView;
	mImmediateContext->OMSetRenderTargets(1, &renderTargetView, NULL);

	UpdateTexture(mouseButtons, mouseX, mouseY);

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
