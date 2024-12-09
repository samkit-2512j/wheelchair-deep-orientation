import argparse as ap
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import norm
from scipy.stats import circmean, circstd
import tensorflow.keras.backend as K

from deep_orientation import beyer    # noqa # pylint: disable=unused-import
from deep_orientation import mobilenet_v2    # noqa # pylint: disable=unused-import
from deep_orientation import beyer_mod_relu     # noqa # pylint: disable=unused-import

from deep_orientation.inputs import INPUT_TYPES
from deep_orientation.inputs import INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)
import deep_orientation.preprocessing as pre
import deep_orientation.postprocessing as post

import utils.img as img_utils
from utils.io import get_files_by_extension

# Constants for input types
INPUT_DEPTH = 'depth'
INPUT_RGB = 'rgb'
INPUT_DEPTH_AND_RGB = 'depth_and_rgb'

# Constants for output types
OUTPUT_BITERNION = 'biternion'
OUTPUT_REGRESSION = 'regression'
OUTPUT_CLASSIFICATION = 'classification'

def _parse_args():
    """Parse command-line arguments"""
    desc = 'Apply neural network for orientation estimation'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('model',
                        type=str,
                        help=("Model to use: beyer, beyer_mod_relu or "
                              "mobilenet_v2"),
                        choices=['beyer', 'beyer_mod_relu', 'mobilenet_v2'])

    parser.add_argument('weights_filepath',
                        type=str,
                        help="Path to the weights to load")

    parser.add_argument('image_or_image_basepath',
                        type=str,
                        help=("Path to a single image file or to a directory "
                              "containing multiple image files"))

    # input -------------------------------------------------------------------
    parser.add_argument('-it', '--input_type',
                        type=str,
                        default=INPUT_DEPTH,
                        choices=INPUT_TYPES,
                        help=(f"Input type. One of {INPUT_TYPES}, default: "
                              f"{INPUT_DEPTH}"))

    parser.add_argument('-iw', '--input_width',
                        type=int,
                        default=46,
                        help="Patch width to use, default: 96")

    parser.add_argument('-ih', '--input_height',
                        type=int,
                        default=46,
                        help="Patch height to use, default: 96")

    parser.add_argument('-ip', '--input_preprocessing',
                        type=str,
                        default='standardize',
                        choices=['standardize', 'scale01', 'none'],
                        help="Preprocessing to apply. One of [standardize, "
                             "scale01, none], default: standardize")

    # output ------------------------------------------------------------------
    parser.add_argument('-n', '--n_samples',
                        type=int,
                        default=1,
                        help="If `n_samples` > 1, dropout sampling is applied,"
                             " default: 1")

    parser.add_argument('-ot', '--output_type',
                        type=str,
                        default=OUTPUT_BITERNION,
                        choices=OUTPUT_TYPES,
                        help=(f"Output type. One of {OUTPUT_TYPES}, default: "
                              f"{OUTPUT_BITERNION})"))

    parser.add_argument('-nc', '--n_classes',
                        type=int,
                        default=8,
                        help=(f"Number of classes when output_type is "
                              f"{OUTPUT_CLASSIFICATION}, default: 8"))

    # other -------------------------------------------------------------------
    parser.add_argument('-ma', '--mobilenet_v2_alpha',
                        type=float,
                        choices=[0.35, 0.5, 0.75, 1.0],
                        default=1.0,
                        help="Alpha value for MobileNet v2 (default: 1.0)")

    parser.add_argument('-d', '--devices',
                        type=str,
                        default='0',
                        help="GPU device id(s) to use. (default: 0)")

    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        default=False,
                        help="CPU only, do not run with GPU support")

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Enable verbose output")

    # return parsed args
    return parser.parse_args()


def load_network(model_name, weights_filepath,
                 input_type, input_height, input_width,
                 output_type,
                 sampling=False,
                 **kwargs):

    # load model --------------------------------------------------------------
    model_module = globals()[model_name]
    model_kwargs = {}
    if model_name == 'mobilenet_v2' and 'mobilenet_v2_alpha' in kwargs:
        model_kwargs['alpha'] = kwargs.get('mobilenet_v2_alpha')
    if output_type == OUTPUT_CLASSIFICATION:
        assert 'n_classes' in kwargs
        model_kwargs['n_classes'] = kwargs.get('n_classes')

    model = model_module.get_model(input_type=input_type,
                                   input_shape=(input_height, input_width),
                                   output_type=output_type,
                                   sampling=sampling,
                                   **model_kwargs)

    # load weights ------------------------------------------------------------
    model.load_weights(weights_filepath)

    return model

def load_and_preprocess(inputs, args):
    # Unpack inputs
    if args.input_type == INPUT_DEPTH_AND_RGB:
        depth_filepath, rgb_filepath = inputs
    elif args.input_type == INPUT_DEPTH:
        depth_filepath, = inputs
    else:
        rgb_filepath, = inputs

    # Pack shape
    shape = (args.input_height, args.input_width)

    # Prepare depth input
    if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        depth = img_utils.load(depth_filepath)
        depth = pre.resize_depth_img(depth, shape)
        depth = depth[..., None]
        depth = pre.preprocess_img(
            depth,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True
        )
        if K.image_data_format() == 'channels_last':
            axes = 'b01c'
        else:
            axes = 'bc01'
        depth = img_utils.dimshuffle(depth, '01c', axes)
        if args.n_samples > 1:
            depth = np.repeat(depth, args.n_samples, axis=0)

    # Prepare RGB input
    if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        rgb = img_utils.load(rgb_filepath)
        rgb = pre.resize_depth_img(rgb, shape)
        rgb = pre.preprocess_img(
            rgb,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True
        )
        if K.image_data_format() == 'channels_last':
            axes = 'b01c'
        else:
            axes = 'bc01'
        rgb = img_utils.dimshuffle(rgb, '01c', axes)
        if args.n_samples > 1:
            rgb = np.repeat(rgb, args.n_samples, axis=0)

    # Return preprocessed images
    if args.input_type == INPUT_DEPTH_AND_RGB:
        return depth, rgb
    elif args.input_type == INPUT_DEPTH:
        return depth,
    else:
        return rgb,

def postprocess(output, args):
    if args.output_type == OUTPUT_BITERNION:
        return post.biternion2deg(output)
    elif args.output_type == OUTPUT_REGRESSION:
        return post.rad2deg(output)
    else:
        return post.class2deg(np.argmax(output, axis=-1), args.n_classes)

def main():
    # Parse args
    args = _parse_args()

    # Set device and data format
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        args.devices = ''
    else:
        args.devices = '/gpu:0'
    K.set_image_data_format(args.data_format)

    # Load the model
    model = load_network(
        model_name=args.model,
        weights_filepath=args.weights,
        input_type=args.input_type,
        input_height=args.input_height,
        input_width=args.input_width,
        output_type=args.output_type,
        sampling=args.sampling,
        n_classes=args.n_classes,
        mobilenet_v2_alpha=args.mobilenet_v2_alpha
    )

    # Collect file paths
    if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        depth_filepaths = get_files_by_extension(
            args.image_or_image_basepath, extension='.png',
            flat_structure=True, recursive=True, follow_links=True
        )
    if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        rgb_filepaths = get_files_by_extension(
            args.image_or_image_basepath, extension='.jpg',
            flat_structure=True, recursive=True, follow_links=True
        )

    # Load and process images
    if args.input_type == INPUT_DEPTH_AND_RGB:
        filepaths = list(zip(depth_filepaths, rgb_filepaths))
    elif args.input_type == INPUT_DEPTH:
        filepaths = list(depth_filepaths)
    else:
        filepaths = list(rgb_filepaths)

    # Initialize visualization
    len_cnt = len(str(len(filepaths)))
    plt.ion()
    fig = plt.figure(figsize=(8, 6))

    for i, inputs in enumerate(filepaths):
        print(f"[{i+1:0{len_cnt}}/{len(filepaths):0{len_cnt}}]: {inputs}")

        # Load and preprocess inputs
        nw_inputs = load_and_preprocess(inputs, args)

        # Predict
        nw_output = model.predict(nw_inputs, batch_size=args.n_samples)

        # Postprocess output
        output = postprocess(nw_output, args)

        # Visualize inputs and predicted angle
        plt.clf()

        # Visualize inputs
        for j, inp in enumerate(nw_inputs):
            # First element of input batch
            img = inp[0]

            # Convert to 01c
            if K.image_data_format() == 'channels_last':
                axes = '01c'
            else:
                axes = 'c01'
            img = img_utils.dimshuffle(img, axes, '01c')

            # Inverse preprocessing
            img = pre.preprocess_img_inverse(
                img,
                scale01=args.input_preprocessing == 'scale01',
                standardize=args.input_preprocessing == 'standardize',
                zero_mean=True,
                unit_variance=True
            )

            # Show
            ax = fig.add_subplot(1, len(nw_inputs) + 1, j + 1)
            if img.shape[-1] == 1:
                ax.imshow(img[:, :, 0], cmap='gray',
                          vmin=img[img != 0].min(), vmax=img.max())
            else:
                ax.imshow(img)
            ax.axis('off')

        # Visualize output
        ax = fig.add_subplot(1, len(nw_inputs) + 1, len(nw_inputs) + 1, polar=True)
        ax.set_theta_zero_location('S', offset=0)
        ax.hist(np.deg2rad(output), width=np.deg2rad(2), density=True,
                alpha=0.5 if args.n_samples > 1 else 1.0, color='#1f77b4')

        if args.n_samples > 1:
            mean_rad = circmean(np.deg2rad(output))
            std_rad = circstd(np.deg2rad(output))
            x = np.deg2rad(np.linspace(0, 360, 360))
            pdf_values = norm.pdf(x, mean_rad, std_rad)
            ax.plot(x, pdf_values, color='#1f77b4', zorder=2, linewidth=2)
            ax.fill(x, pdf_values, color='#1f77b4', zorder=2, alpha=0.3)

        ax.set_yscale('symlog')
        ax.set_ylim([0, 20])

        plt.tight_layout()
        plt.pause(0.0005)

if __name__ == "__main__":
    main()
