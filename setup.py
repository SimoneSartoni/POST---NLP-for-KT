import numpy
from setuptools import setup, Extension

module = Extension('Compute_Similarity_Cython', sources=['C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/code/utils/Compute_Similarity_Cython.pyx'],
                   include_dirs=[numpy.get_include()])


setup(
    name='NLPforKT',
    version='1.0',
    packages=['Knowledge_Tracing', 'Knowledge_Tracing.code', 'Knowledge_Tracing.code.utils',
              'Knowledge_Tracing.code.Similarity', 'Knowledge_Tracing.code.Similarity.Cython'],
    author='Simone Sartoni',
    author_email='10583763@polimi.it',
    description='Cython compile',
    ext_modules=[module]
)
