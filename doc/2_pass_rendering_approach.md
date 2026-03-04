## problems of current approach

the current approach bounces the ray into jittered photon souces directions only and comes with several disadvantages:
1. butterflies in the end (unclear, but probably the guided path hits a light source mostly even in dark areas)
2. no convergence: probably related to 1.
3. moderately slow (searching photons in the 3x3x3 grid)

We need to completely revise the approach. my proposal is a 2-pass approach:

1. photon gathering pass that stops on 1st camera hit
2. guided path tracing

## evaluate photons on first camera hit

currently we are using photons for directions of guided path tracing only. this neglects the fact that we already have precious information about the indirect lighting. 

the idea is to sample neighboured photons on first hit and to not follow up with bounces.
this is the classical approach where neighbored photons are aggregated to lighting information. this needs to be physically correct (NEE, BSDF, PDF, MIS...)

there is two versions of this approach:
1. diffuse lighting pass: gather photons in bigger neighborhood, all non-delta photons (diffuse lighting)
2. caustic lighting pass: gather photons in small neighborhood, all delta photons (for caustics etc.)

these two passes are done ONCE per photon map update. they are done just once after the update: 
running them more than once on the same photon map does not bring any value. 
they are ran before the guided path tracing step.

they use the existing dense grid, but if that causes artifacts, we might need to introduce hash grid again (I want to avoid).
all filters for photon gathering must be respected DEPENDING ON THE HITPOINT MATERIAL (positive halfspace for non-translucent materials, tau, photon normal direction...etc)

we currently do not have a re-computation of the photon map, but your design can already prepare for this!

 ## guided path tracing for refinement

 the first two passes are then continouosly refine by guided path tracing. 

 I have trust in the current approach that 
 1. gathers photons in the hitpoint cell or 3x3x3 neighborhood
 2. filters (positive halfspace for non-translucent materials, tau, photon normal direction...etc) and picks a random
 3. bounces the ray into a cosine-weighted jittered source direction of that photon
however, we learned that this approach produces butterflies and does not converg well.

the idea to improve is:

 ### balance between jittered guidance and full (hemi-)sphere

 currently we focus the bounce on photon source directions ONLY. I think the better approach is to also bounce into the ray into other directions of the hemisphere (non-delta) or sphere (delta).
 
 there could be a ratio between 
 a) jittered photon source direction and
 b) random direction (russian roulette)
 of 50% each. 

 IMPORTANT: bounces have to process the FULL material (BSDF) complexity: that means for gathering the random photon, we have to differentiate between

 1. delta materials (eg. glass): ray bounces into full sphere directions. this impacts the photon filtering (positive half-space does not apply etc.). 
 2. non-delta materials: ray bounces into positive hemisphere only (positive halfspace for non-translucent materials, tau, photon normal direction...etc)

 we can use the photon tag of the randomly picked photon to identify these two cases!

 russian roulette after a certain bounce depth can be removed for this approach: it's inherently already part of each bounce.

## the values

- keep code simple and straightforward. avoid template code.
- clean up existing code that is not needed anymore

 ## your tasks

 - Review and compare with academic approaches


 # updates

## 1
 - during the final rendering, I want to see the accumulated rendering of pass1 + pass2, updated after each iteration. this request does NOT imply that pass 1 must  be computed more than once.but it populates the window first, and then pass 2 contiously contributes
- add detailed and relevant debug information to the .json output about the 2 passes
- the gamma of  the .exr HDR file seemed off: it needs to be linear color space as far as I know
 

