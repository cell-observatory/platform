
Template for docker/apptainer containers
====================================================

- [Docker \& Apptainer images](#docker--apptainer-images)
- [Clone repository to your host system](#clone-repository-to-your-host-system)
- [Running docker image](#running-docker-image)


# Docker [images](https://github.com/cell-observatory/platform/pkgs/container/platform)
Our prebuilt image with Python, Torch, and all packages installed for you.
```shell
docker pull ghcr.io/cell-observatory/platform:main_torch_cuda_12_8
```

# Clone repository to your host system
```shell
git clone --recurse-submodules https://github.com/cell-observatory/platform.git
```

To later update to the latest, greatest
```shell
git pull --recurse-submodules
```

**Note:** If you want to run a local version of the image, see the [Dockerfile](https://github.com/cell-observatory/platform/blob/main/Dockerfile)


# Running docker image

To run docker image, cd to repo directory or replace `$(pwd)` with your local path for the repository.
```shell
docker run --network host -u 1000 --privileged -v $(pwd):/app/platform -w /app/platform --env PYTHONUNBUFFERED=1 --pull missing -it --rm  --ipc host --gpus all ghcr.io/cell-observatory/platform:main_torch_cuda_12_8 bash
```

# Running docker image on a cluster via Apptainer

Running an image on a cluster typically requires an Apptainer version of the image, which can be generated by:
```shell
apptainer build --nv --force develop_torch_cuda_12_8.sif docker://ghcr.io/cell-observatory/platform:develop_torch_cuda_12_8
```
**Note:** Replace `docker-daemon` with your local docker image.
