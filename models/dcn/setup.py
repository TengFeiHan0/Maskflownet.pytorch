#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

def make_cuda_ext(name, sources, sources_cuda=[]):
    
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{name}',
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

setup(
    name='deform_conv_ext',
    ext_modules=[
        make_cuda_ext(
                name='deform_conv_ext',
                sources=['deform_conv_ext.cpp'],
                sources_cuda=[
                    'deform_conv_cuda.cpp',
                    'deform_conv_cuda_kernel.cu'
                ])     
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
