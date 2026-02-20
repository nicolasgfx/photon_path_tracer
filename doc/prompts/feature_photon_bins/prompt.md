sounds good! I want to combine ALL your proposals with this strategy. analyse my strategy, estimate the impact on the code and the impact on the speed and quality.

the idea: we create [n] lobes (bins) for each pixel instead of processing all 500+ photons in a loop. the bins are very similar to idea (B1). a bin represents a specrum im incoming photon directions. the bin are static and pre-defined: they are just "populated" with the photon information (incoming direction, flux...) of the cached cells of a pixel. the motivation is that the 16 samples bounce in directions where light actually comes from, rather than random.

the bins have to cover a sphere, not only a hemisphere (for caustics, glass), but for diffuce bounces, only the bins on the positive hemisphere must be considered. since the bins are a discrete approximation of the hemisphere, we have to consider the edge case that most light comes from a bin that is partially in the negative half space of a pixel's normal (surface point). we do this by accepting this lobe, but the bounced ray must be in the positive halfspace.

benefits of lobes/bins:
- significant reduction of per-pixel lookups
- the per-pixel buffer (A1) can be a straightforward widthxheightx[n] array for simplicity and cache optimization
- optional: in the GPU kernel , we can unroll the loop into [n] static lookups of the [n] bins. [n] would be somewhere around 16 or 32. since the shader code is not dynamic, we create >1 shader kernels, where each kernel covers a different amount of bins (16, 32, 64...). the kernel is determined by a flag in "config.h"
- (B1) we use the bins as source of truth for the  16 samples to bounce in directions that are informed by where light actually comes from, rather than random. BUT, we have to jitter within one bin direction!!
- (B2): we use the bins for the NEE shadow ray directions. for this, bins have to be sorted (!) in some way, so that we can easily identify the most relevant bin
- (B3) yes, improve the SPP


write a DETAILED coding plan for my proposal into "coding_plan.md", including
- your review
- comparison with existing approaches
- impact on our speed and quality
- diagram of "old bounce" vs "new bounce" strategy
- changes to the code
- unittest to be written
- files to be changed

