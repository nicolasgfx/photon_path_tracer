#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define TINYEXR_USE_MINIZ (0)
#define TINYEXR_USE_STB_ZLIB (1)
#define TINYEXR_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4245 4702)
#endif
#include "tinyexr.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
