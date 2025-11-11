# DTS Algorithmics

This is the repo for the algorithms part of our DTS system:

<p align="center">DTS = Drone Tracking System</p>

Our goal was to estimate the direction of audio sources (specifically drones) with a small circular microphone array, as part of the undergraduate engineering course "Signals and Systems".

Previously we implemented the `Music` algorithm as well, but decided to use our own as it featured predominately the mathematical tools acquired in the course.

The algorithm works by interpolating the cross-correlation between pairs of microphones to estimate the `TDOA`:

$$
\tau_{ij} = \arg\max_{\tau} \int_{-\infty}^{\infty} x_i^{*}(t')\, x_j(t' + \tau)\, dt'
$$

We then take $\tau_{i0}$ from each microphone $i$ to the array center, interpolate these delays as a function of the known microphone angles, and choose the azimuthal angle $\hat\theta$ that minimizes the interpolated TDOA:

$$
\hat{\theta} = \arg\min_{\theta} \tau(\theta)
$$

Using the geometric positions of the microphones placed on the outer circle, one can then use this to estimate the polar angle $\hat\varphi$ by averaging the estimated polar angles from all mic pairs ($|\Omega|=\frac{N(N-1)}{2}$ pairs for $N$ mics):

$$
\hat\varphi = \frac{1}{|\Omega|} \sum_{(i,j)\in\Omega} \arccos\left(\frac{c\cdot\tau_{ij}}{d_{ij}\cdot cos(\frac{\theta_i+\theta_j}{2}-\hat\theta)}\right)
$$

with

$$
d_{ij} = 2R\cdot sin\left(\frac{\Delta\theta_{ij}}{2}\right)
$$

where $d_{ij}$ is the distance between the respective microphone pairs $(i,j)$, $\Delta\theta_{ij}$ is the difference between the two microphone angles, R is the mic array radius, and c is the speed of sound.

for simplicity, we only used the center microphone as reference, so the formula changes slightly. The final polar angle estimate is:

$$
\hat\varphi = \frac{1}{N-1} \sum_{i=1}^{N-1} \arccos\left(\frac{c\cdot\tau_{i0}}{R\cdot cos(\theta_i-\hat\theta)}\right)
$$

All of these expressions are calculated digitally, accounting for the fact that the cross-correlations are computed on a finite time window and not $[-\infty,\infty]$, etc.

Each estimated coordinate pair $(\hat\varphi,\hat\theta)$ is added to a queue and filtered to account for noise.

# Robotics

The control code for the robot itself was written in C++ and is in a different repository.


