# Test results

Tone-mapped PSNR in dB (higher is better) on real noisy frames from two of the author's cameras. The frames were never used in training. Every denoiser runs through Chromaforge's model contract on the same 512-cell tile. Its output is binned back to one RGB value per Bayer cell and compared with the aligned, brightness-matched base-ISO reference. Cells where something moved between frames are masked out. The best denoiser in each row is in bold.

These scenes are the author's own photos, so they are not published. The scripts that produce these numbers are `training/real_pairs.py` and `training/compare_cf.py`.

At ISO 200–400 the *Noisy* column scores highest. That is a limit of the test, not of the denoisers: the reference is the mean of only 2–5 base-ISO frames, so it carries about as much noise as a low-ISO test frame. Read those rows only to compare the denoisers with each other.

## Sony A7 IV, centre tile

| ISO | Noisy | Built-in | LightCraft | RawNIND | ChromaClean 0.2.0 | ChromaClean 0.3.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 48.04 | 43.97 | 46.17 | 44.87 | **46.35** | 46.31 |
| 400 | 44.18 | 42.49 | 44.40 | 43.45 | 44.62 | **44.62** |
| 800 | 41.45 | 41.87 | 43.29 | 42.70 | 43.51 | **43.54** |
| 1600 | 38.86 | 41.21 | 42.44 | 42.13 | 42.69 | **42.75** |
| 3200 | 35.81 | 40.02 | 41.03 | 40.93 | 41.31 | **41.35** |
| 6400 | 32.98 | 39.19 | 40.13 | 40.17 | 40.52 | **40.52** |
| 12800 | 30.02 | 38.28 | 39.48 | 39.46 | 39.81 | **39.81** |
| 25600 | 26.72 | 36.49 | 38.26 | 38.21 | **38.52** | 38.50 |

## Sony A7 IV, most detailed tile

| ISO | Noisy | Built-in | LightCraft | RawNIND | ChromaClean 0.2.0 | ChromaClean 0.3.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 46.72 | 40.51 | 43.03 | 41.19 | **43.09** | 42.91 |
| 400 | 41.84 | 38.38 | 40.57 | 39.05 | **40.64** | 40.51 |
| 800 | 39.33 | 38.21 | 39.45 | 38.32 | **39.58** | 39.45 |
| 1600 | 37.11 | 37.49 | 38.69 | 37.74 | **38.83** | 38.73 |
| 3200 | 34.28 | 35.85 | 37.12 | 36.46 | **37.24** | 37.22 |
| 6400 | 32.07 | 34.90 | 36.37 | 35.78 | **36.46** | 36.45 |
| 12800 | 29.32 | 33.77 | 34.74 | 34.12 | **34.83** | 34.80 |
| 25600 | 26.40 | 32.23 | 33.87 | 33.41 | **33.99** | 33.94 |

## Olympus E-M10 IV, centre tile

| ISO | Noisy | Built-in | LightCraft | RawNIND | ChromaClean 0.2.0 | ChromaClean 0.3.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 40.39 | 36.67 | 38.29 | 36.30 | 38.36 | **38.44** |
| 800 | 36.93 | 36.93 | 37.95 | 36.57 | 37.98 | **38.02** |
| 1600 | 34.28 | 36.81 | 37.46 | 36.22 | 37.50 | **37.55** |
| 3200 | 31.49 | 35.63 | 36.38 | 35.26 | 36.40 | **36.44** |
| 6400 | 27.98 | 34.32 | 35.18 | 34.22 | 35.21 | **35.28** |
| 12800 | 23.83 | 33.10 | 34.71 | 33.65 | 34.76 | **34.83** |
| 25600 | 19.37 | 31.84 | 34.93 | 33.57 | 35.02 | **35.13** |

## Olympus E-M10 IV, most detailed tile

| ISO | Noisy | Built-in | LightCraft | RawNIND | ChromaClean 0.2.0 | ChromaClean 0.3.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 41.50 | 35.72 | 38.28 | 35.36 | **38.42** | 38.31 |
| 800 | 38.65 | 37.12 | 38.83 | 37.03 | **38.92** | 38.82 |
| 1600 | 36.42 | 37.01 | 38.10 | 36.56 | **38.21** | 38.16 |
| 3200 | 33.93 | 35.77 | 37.17 | 35.72 | **37.23** | 37.20 |
| 6400 | 30.87 | 34.60 | 36.19 | 34.83 | 36.18 | **36.20** |
| 12800 | 26.89 | 32.56 | **34.84** | 33.37 | 34.78 | 34.84 |
| 25600 | 22.53 | 31.73 | **35.00** | 33.07 | 34.89 | 34.93 |

Built-in is Chromaforge's block-matching denoiser. LightCraft and RawNIND are the AI denoise models the app offers, run with their own manifests' settings.
