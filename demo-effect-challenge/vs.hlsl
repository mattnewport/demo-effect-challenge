cbuffer VSConstants {
	float2 offset;
}

float4 main(float4 pos : POSITION) : SV_POSITION
{
	return float4(pos.xy + offset, pos.z, pos.w);
}