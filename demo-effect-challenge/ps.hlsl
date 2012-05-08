texture2D tex : register(t0);

SamplerState samplerState : register(s0);

cbuffer PSConstants {
	float2 screenSize;
	float2 texSize;
}

float4 main(float4 pos : SV_POSITION) : SV_TARGET
{
	return tex.Sample(samplerState, float2(pos.x / texSize.x, pos.y / texSize.y));
}