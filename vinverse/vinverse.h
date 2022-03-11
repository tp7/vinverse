#pragma once

#include "avisynth.h"

enum class VinverseMode
{
    Vinverse,
    Vinverse2
};

class Vinverse : public GenericVideoFilter
{
public:
    Vinverse(PClip child, float sstr, int amnt, int uv, float scl, int opt, VinverseMode mode, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

    int __stdcall SetCacheHints(int cachehints, int frame_range) override
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    ~Vinverse();

private:
    float sstr_;
    float scl_;
    int amnt_;
    int uv_;
    VinverseMode mode_;

    uint8_t* blur3_buffer, * blur6_buffer;
    int* dlut;

    int pb_pitch;
    uint8_t* buffer;

    bool sse2;
    bool v8;

    void(*blur3)(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height);
    void(*blur5)(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height);
    void(*sbr)(uint8_t* dstp, uint8_t* tempp, const uint8_t* srcp, int dst_pitch, int temp_pitch, int src_pitch, int width, int height);
    void(*fin_plane)(uint8_t* dstp, const uint8_t* srcp, const uint8_t* pb3, const uint8_t* pb6, const int* dlut, int dst_pitch, int src_pitch, int pb_pitch, int width, int height, int amnt);
};

void vertical_blur3_sse2(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height);
void vertical_blur5_sse2(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height);
void vertical_sbr_sse2(uint8_t* dstp, uint8_t* tempp, const uint8_t* srcp, int dst_pitch, int temp_pitch, int src_pitch, int width, int height);
void finalize_plane_sse2(uint8_t* dstp, const uint8_t* srcp, const uint8_t* pb3, const uint8_t* pb6, float sstr, float scl, int src_pitch, int dst_pitch, int pb_pitch, int width, int height, int amnt);
