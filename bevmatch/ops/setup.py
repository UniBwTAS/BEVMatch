"""Builds the two CUDA extensions used by the BEV backbone.

    cd bevmatch/ops && python setup.py build_ext --inplace

`voxel_layer` provides hard/dynamic voxelization and scatter operations, `bev_pool_ext`
the pooling that lifts camera features into the BEV grid. Both are taken from BEVFusion
and need a CUDA toolkit matching the installed PyTorch build.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bevmatch_ops',
    ext_modules=[
        CUDAExtension(
            name='voxel.voxel_layer',
            sources=[
                'voxel/src/voxelization.cpp',
                'voxel/src/voxelization_cpu.cpp',
                'voxel/src/voxelization_cuda.cu',
                'voxel/src/scatter_points_cpu.cpp',
                'voxel/src/scatter_points_cuda.cu',
            ],
        ),
        CUDAExtension(
            name='bev_pool.bev_pool_ext',
            sources=[
                'bev_pool/src/bev_pool.cpp',
                'bev_pool/src/bev_pool_cuda.cu',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
