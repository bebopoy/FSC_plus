from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = "pointnet2_ops/_ext-src"
_ext_sources = [
    osp.join(_ext_src_root, "src", f) for f in [
        "ball_query.cpp",
        "ball_query_gpu.cu",
        "bindings.cpp",
        "group_points.cpp",
        "group_points_gpu.cu",
        "interpolate.cpp",
        "interpolate_gpu.cu",
        "sampling.cpp",
        "sampling_gpu.cu"
    ]
]

setup(
    name="pointnet2_ops",
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-Xfatbin",
                    "-compress-all",
                    "-gencode=arch=compute_50,code=sm_50",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86"
                ],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)