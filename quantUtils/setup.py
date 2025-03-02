from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='quantUtils',
    ext_modules=[
        CUDAExtension(
            'quantUtils',
            ['bfp_quant_util.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
