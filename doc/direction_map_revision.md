## the problem

- the dense grid does not perform well in the knn photon search
- replace dense grid with hash grid. goal is to reliably find knn photons. 
- hard requirement for hash grid needs is to support multiple photons per cm (0.01f). collisions must be under 5%

## Changes to current Pass 1

- in pass 1 (camera 1st hit), gather knn photons in a certain radius around the hitpoint. 
- the novelty is that shadow rays are trace from the hitpoint to all neighbored photons. shadow ray intersections remove the need for tau parameters, positive half space checks, normal checks etc.
- the out

## shadow rays intersection rules

- the shadow ray pass is another filter on the knn photon search
- since the surface hit point itself is on a triangle surface, self-intersections must be avoided. 
- each shadow ray starts at the surface hitpoint, and is shot towards one of the knn photons sequentially

Photon Acceptance Rules:
- if the shadow ray hits a non-delta triangle before the photon is reached, we reject this photon
- if the shadow ray hits a delta triangle (glass, glossy, ...) before the photon is reached, we accept this photon
- if the shadow ray hits a translucent triangle before the photon is reached, we accept this photon
- if the shadow ray hits a light source triangle before the photon is reached, we reject this photon
- if the shadow ray hits nothing and the photon is reached, we accept this photon

## photon weighting
 
- all filteres photons are added to the fibonacci sphere with a default weight "w = 1"
- shadow rays that hit a delta triangle get a weight of w=4
- shadow rays that hit a translucent triangle get a weight of w=4

## fibonacci sphere

- no changes to the current approach with 128 bins