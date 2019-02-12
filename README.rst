.. Variables to use the correct hyperlinks in the readmertd build
.. |classifier| replace:: `morpheus.classifier.Classifier <https://morpheus-astro.readthedocs.io/en/latest/source/morpheus.html#morpheus.classifier.Classifier>`__
.. |classify_arrays| replace:: `classify_arrays <https://morpheus-astro.readthedocs.io/en/latest/source/morpheus.html#morpheus.classifier.Classifier.classify_arrays>`__
.. |classify_files| replace:: `classify_files <https://morpheus-astro.readthedocs.io/en/latest/source/morpheus.html#morpheus.classifier.Classifier.classify_files>`__
.. |catalog_arrays| replace:: `catalog_arrays <https://morpheus-astro.readthedocs.io/en/latest/source/morpheus.html#morpheus.classifier.Classifier.catalog_arrays>`__

.. .. image:: https://cdn.jsdelivr.net/gh/morpheus-project/morpheus/morpheus.png
    :target: https://github.com/morpheus-project/morpheus
    :align: center

########
Morpheus
########

====================================

.. image:: https://travis-ci.com/morpheus-project/morpheus.svg?branch=master
    :target: https://travis-ci.com/morpheus-project/morpheus

.. image:: https://codecov.io/gh/morpheus-project/morpheus/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/morpheus-project/morpheus

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://www.python.org/downloads/release/python-360/

.. image:: https://readthedocs.org/projects/morpheus-astro/badge/?version=latest
    :target: https://morpheus-astro.readthedocs.io

Morpheus is a neural network model used to generate pixel level morphological
classifications for astronomical sources. This model can be used to generate
segmentation maps or to inform other photometric measurements with granular
morphological information.

Installation
============

Morpheus is implemented using `TensorFlow <https://www.tensorflow.org/>`_.
TensorFlow is **not** listed in the dependencies for the package. So you need
to install TensorFlow before you install Morpheus. It has to be done this way
to support the GPU accelerated version of TensorFlow, which has a different
package name. For more information on installing TensorFlow visit the
`TensorFlow website <https://www.tensorflow.org/install/>`_.


.. code-block:: bash

    pip install morpheus-astro

Docker
======

Morpheus has two main flavors of Docker Image: ``gpu`` for the GPU enabled version
of TensorFlow and ``cpu`` for the standard CPU implementation of TensorFlow.
Visit the `Docker Hub <https://hub.docker.com/r/morpheusastro/morpheus>`_ page
for relevant tags.

For GPU support:

.. code-block:: bash

    docker run --runtime=nvidia -it morpheusastro/morpheus:0.3-gpu

For CPU only:

.. code-block:: bash

    docker run -it morpheusastro/morpheus:0.3-cpu


Usage
=====

The main way to interact with Morpheus is by using the |classifier|
class. Using this class you can classify astronomical images in 2 ways:

1. Using |classify_arrays| to classify numpy arrays.

.. code-block:: python

    from morpheus.classifier import Classifier
    from morpheus.data import example

    h, j, v, z = example.get_sample()
    morphs = Classifier.classify_arrays(h=h, j=j, v=v, z=z)

The output that is returned is a dictionary where the keys are the
morphological classes: spheroid, disk, irregular, point source, and background
and the values are the corresponding numpy arrays.

2. Using |classify_files| to classify FITS files:

.. code-block:: python

    from morpheus.classifier import Classifier
    from morpheus.data import example

    # this saves the sample numpy arrays as FITS files in 'out_dir'
    example.get_sample(out_dir='.')
    h, j, v, z = [f'{band}.fits' for band in 'hjvz']

    morphs = Classifier.classify_files(h=h, j=j, v=v, z=z)

Using FITS files can be useful for classifying files that are too large to fit
into memory. If an image is too large to fit into memory, then specify the
``out_dir`` argument and the outputs will be saved there rather than returned.

.. code-block:: python

    from morpheus.classifier import Classifier
    from morpheus.data import example

    # this saves the sample numpy arrays as fits files in 'out_dir'
    example.get_sample(out_dir='.')
    h, j, v, z = [f'{band}.fits' for band in 'hjvz']

    Classifier.classify_files(h=h, j=j, v=v, z=z, out_dir='.')

If you're classifying a large image and have multiple NVIDIA GPUs on the same
machine available the image can be classified in parallel using the ``gpus``
argument. The image split evenly along the first axis and then handed off
to subprocess to classify the subset of the image, after which, the image is
stitched back together.

.. code-block:: python

    from morpheus.classifier import Classifier

    # h, j, v, and, z are strings that point to a large image

    # gpus should be an integer list containing the GPU ids for the GPUs that
    # you want to use to classify the images. You can get these values by
    # calling 'nvidia-smi'
    gpus = [0, 1]

    Classifier.classify_files(h=h, j=j, v=v, z=z, out_dir='.', gpus=gpus)

Morpheus can also be used to generate a catalog of morphological
classifications from an input image using |catalog_arrays|. The catalog will
include an *id*, *location* (in pixels), and a *morphological classification*
for each source in the following order: [Spheroid, Disk, Irregular, Point
Source].

.. code-block:: python

    from morpheus.classifier import Classifier
    from morpheus.data import example

    h, j, v, z = example.get_sample()
    catalog = Classifier.catalog_arrays(h=h, j=j, v=v, z=z)

Demo
=============
Try it out on `Google Colab <https://colab.research.google.com/github/morpheus-project/morpheus/blob/master/examples/example_array.ipynb>`_!


Documentation
=============

https://morpheus-astro.readthedocs.io/
