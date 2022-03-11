#include <algorithm>
#include <emmintrin.h>

#include "vinverse.h"

#ifdef _MSC_VER 
#define WIN32_LEAN_AND_MEAN
#endif

AVS_FORCEINLINE void* aligned_malloc(size_t size, size_t align)
{
    void* result = [&]() {
#ifdef _MSC_VER 
        return _aligned_malloc(size, align);
#else 
        if (posix_memalign(&result, align, size))
            return result = nullptr;
        else
            return result;
#endif
    }();

    return result;
}

AVS_FORCEINLINE void aligned_free(void* ptr)
{
#ifdef _MSC_VER 
    _aligned_free(ptr);
#else 
    free(ptr);
#endif
}

static void vertical_blur3_c(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* srcpp = y == 0 ? srcp + src_pitch : srcp - src_pitch;
        const uint8_t* srcpn = y == height - 1 ? srcp - src_pitch : srcp + src_pitch;

        for (int x = 0; x < width; ++x)
            dstp[x] = (srcpp[x] + (srcp[x] << 1) + srcpn[x] + 2) >> 2;

        srcp += src_pitch;
        dstp += dst_pitch;
    }
}

static void vertical_blur5_c(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* srcppp = y < 2 ? srcp + src_pitch * 2 : srcp - src_pitch * 2;
        const uint8_t* srcpp = y == 0 ? srcp + src_pitch : srcp - src_pitch;
        const uint8_t* srcpn = y == height - 1 ? srcp - src_pitch : srcp + src_pitch;
        const uint8_t* srcpnn = y > height - 3 ? srcp - src_pitch * 2 : srcp + src_pitch * 2;

        for (int x = 0; x < width; ++x)
            dstp[x] = (srcppp[x] + ((srcpp[x] + srcpn[x]) << 2) + srcp[x] * 6 + srcpnn[x] + 8) >> 4;

        srcp += src_pitch;
        dstp += dst_pitch;
    }
}

static void mt_makediff_c(uint8_t* dstp, const uint8_t* c1p, const uint8_t* c2p, int dst_pitch, int c1_pitch, int c2_pitch, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
            dstp[x] = std::max(std::min(c1p[x] - c2p[x] + 128, 255), 0);

        dstp += dst_pitch;
        c1p += c1_pitch;
        c2p += c2_pitch;
    }
}

static void vertical_sbr_c(uint8_t* dstp, uint8_t* tempp, const uint8_t* srcp, int dst_pitch, int temp_pitch, int src_pitch, int width, int height)
{
    vertical_blur3_c(tempp, srcp, temp_pitch, src_pitch, width, height); //temp = rg11
    mt_makediff_c(dstp, srcp, tempp, dst_pitch, src_pitch, temp_pitch, width, height); //dst = rg11D
    vertical_blur3_c(tempp, dstp, temp_pitch, dst_pitch, width, height); //temp = rg11D.vblur()

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int t = dstp[x] - tempp[x];
            int t2 = dstp[x] - 128;
            if (t * t2 < 0)
                dstp[x] = srcp[x];
            else
            {
                if (std::abs(t) < std::abs(t2))
                    dstp[x] = srcp[x] - t;
                else
                    dstp[x] = srcp[x] - dstp[x] + 128;
            }
        }
        dstp += dst_pitch;
        srcp += src_pitch;
        tempp += temp_pitch;
    }
}

template <bool amnt_255>
static void finalize_plane_c(uint8_t* dstp, const uint8_t* srcp, const uint8_t* pb3, const uint8_t* pb6, const int* dlut, int dst_pitch, int src_pitch, int pb_pitch, int width, int height, int amnt)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int d1 = srcp[x] - pb3[x] + 255;
            const int d2 = pb3[x] - pb6[x] + 255;
            const int df = pb3[x] + dlut[(d1 << 9) + d2];

            int minm, maxm;
            if (amnt_255)
            {
                minm = 0;
                maxm = 255;
            }
            else
            {
                minm = std::max(srcp[x] - amnt, 0);
                maxm = std::min(srcp[x] + amnt, 255);
            }

            if (df <= minm)
                dstp[x] = minm;
            else if (df >= maxm)
                dstp[x] = maxm;
            else
                dstp[x] = df;
        }

        srcp += src_pitch;
        pb3 += pb_pitch;
        pb6 += pb_pitch;
        dstp += dst_pitch;
    }
}

Vinverse::Vinverse(PClip child, float sstr, int amnt, int uv, float scl, int opt, VinverseMode mode, IScriptEnvironment* env)
    : GenericVideoFilter(child), sstr_(sstr), amnt_(amnt), uv_(uv), scl_(scl), opt_(opt), mode_(mode), blur3_buffer(nullptr), blur6_buffer(nullptr), dlut(nullptr)
{
    if (!vi.IsPlanar())
        env->ThrowError("Vinverse: only planar input is supported!");
    if (amnt < 1 || amnt > 255)
        env->ThrowError("Vinverse: amnt must be greater than 0 and less than or equal to 255!");
    if (uv < 1 || uv > 3)
        env->ThrowError("Vinverse: uv must be set to 1, 2, or 3!");
    if (opt_ < -1 || opt_ > 1)
        env->ThrowError("Vinverse: opt must be between -1..1.");

    pb_pitch = (vi.width + 15) & ~15;

    sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);

    if (!sse2 && opt_ == 1)
        env->ThrowError("Vinverse: opt=1 requires SSE2.");

    size_t pbuf_size = vi.height * pb_pitch;
    size_t dlut_size = 512 * 512 * sizeof(int);

    buffer = reinterpret_cast<uint8_t*>(aligned_malloc(pbuf_size * 2 + dlut_size, 16));

    if (buffer == nullptr)
        env->ThrowError("Vinverse:  malloc failure!");

    blur3_buffer = buffer;
    blur6_buffer = blur3_buffer + pbuf_size;

    dlut = reinterpret_cast<int*>(blur6_buffer + pbuf_size);

    for (int x = -255; x <= 255; ++x)
    {
        for (int y = -255; y <= 255; ++y)
        {
            const float y2 = y * sstr;
            const float da = fabs(static_cast<float>(x)) < fabs(y2) ? x : y2;
            dlut[((x + 255) << 9) + (y + 255)] = static_cast<float>(x) * y2 < 0.0 ? static_cast<int>(da * scl) : static_cast<int>(da);
        }
    }

    if ((sse2 && opt_ < 0) || opt_ == 1)
    {
        blur3 = vertical_blur3_sse2;
        blur5 = vertical_blur5_sse2;
        sbr = vertical_sbr_sse2;
    }
    else
    {
        blur3 = vertical_blur3_c;
        blur5 = vertical_blur5_c;
        sbr = vertical_sbr_c;
    }

    v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { v8 = false; }

    if (amnt_ == 255)
        fin_plane = finalize_plane_c<true>;
    else
        fin_plane = finalize_plane_c<false>;
}

Vinverse::~Vinverse()
{
    aligned_free(buffer);
}

PVideoFrame __stdcall Vinverse::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = (v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    int planes[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    for (int pid = 0; pid < 3; ++pid)
    {
        int current_plane = planes[pid];
        if (current_plane != PLANAR_Y && (vi.IsY8() || uv_ == 1))
            continue;

        const uint8_t* srcp = src->GetReadPtr(current_plane);
        const int src_pitch = src->GetPitch(current_plane);
        const int height = src->GetHeight(current_plane);
        const int width = src->GetRowSize(current_plane);
        uint8_t* dstp = dst->GetWritePtr(current_plane);
        const int dst_pitch = dst->GetPitch(current_plane);

        if (current_plane != PLANAR_Y && uv_ == 2)
        {
            env->BitBlt(dstp, dst_pitch, srcp, src_pitch, width, height);
            continue;
        }

        if (mode_ == VinverseMode::Vinverse)
        {
            blur3(blur3_buffer, srcp, pb_pitch, src_pitch, width, height);
            blur5(blur6_buffer, blur3_buffer, pb_pitch, pb_pitch, width, height);
        }
        else
        {
            if (current_plane == PLANAR_Y)
                sbr(blur3_buffer, blur6_buffer, srcp, pb_pitch, pb_pitch, src_pitch, width, height);
            else
                env->BitBlt(blur3_buffer, pb_pitch, srcp, src_pitch, width, height);
            blur3(blur6_buffer, blur3_buffer, pb_pitch, pb_pitch, width, height);
        }

        if ((sse2 && opt_ < 0) || opt_ == 1)
            finalize_plane_sse2(dstp, srcp, blur3_buffer, blur6_buffer, sstr_, scl_, src_pitch, dst_pitch, pb_pitch, width, height, amnt_);
        else
            fin_plane(dstp, srcp, blur3_buffer, blur6_buffer, dlut, dst_pitch, src_pitch, pb_pitch, width, height, amnt_);
    }

    return dst;
}

AVSValue __cdecl Create_Vinverse(AVSValue args, void*, IScriptEnvironment* env)
{
    enum { CLIP, SSTR, AMNT, UV, SCL, OPT };
    return new Vinverse(args[CLIP].AsClip(), args[SSTR].AsFloatf(2.7f), args[AMNT].AsInt(255), args[UV].AsInt(3), args[SCL].AsFloatf(0.25f), args[OPT].AsInt(-1), VinverseMode::Vinverse, env);
}

AVSValue __cdecl Create_Vinverse2(AVSValue args, void*, IScriptEnvironment* env)
{
    enum { CLIP, SSTR, AMNT, UV, SCL, OPT };
    return new Vinverse(args[CLIP].AsClip(), args[SSTR].AsFloatf(2.7f), args[AMNT].AsInt(255), args[UV].AsInt(3), args[SCL].AsFloatf(0.25f), args[OPT].AsInt(-1), VinverseMode::Vinverse2, env);
}

const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("vinverse", "c[sstr]f[amnt]i[uv]i[scl]f[opt]i", Create_Vinverse, 0);
    env->AddFunction("vinverse2", "c[sstr]f[amnt]i[uv]i[scl]f[opt]i", Create_Vinverse2, 0);
    return "Doushimashita?";
}
