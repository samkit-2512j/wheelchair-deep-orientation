# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import os
import rospy  # ROS Python library
from std_srvs.srv import Trigger, TriggerResponse  # ROS service types
from std_srvs.srv import Empty  # Service type for /pose/continue

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

import time

# seaborn.set_style('darkgrid')
seaborn.set_context('notebook', font_scale=1.2)


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
    """
    Load and preprocess the image inputs.

    Args:
        inputs: Tuple of file paths (depth, rgb, mask)
        args: Parsed command-line arguments

    Returns:
        Tuple of preprocessed images
    """
    # unpack inputs
    if args.input_type == INPUT_DEPTH_AND_RGB:
        depth_filepath, rgb_filepath, mask_filepath = inputs
    elif args.input_type == INPUT_DEPTH:
        depth_filepath, mask_filepath = inputs
    else:
        rgb_filepath, mask_filepath = inputs

    # pack shape
    shape = (args.input_height, args.input_width)

    # load mask
    mask_filepath = mask_filepath + "cropped_image_1_mask.png"
    
    print(mask_filepath)
    mask = img_utils.load(mask_filepath)
    mask_resized = pre.resize_mask(mask, shape)
    mask_resized = mask_resized > 0

    # prepare depth input
    if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        # load
        depth = img_utils.load(depth_filepath)

        # create mask
        # mask = depth > 0
        # mask_resized = pre.resize_mask(mask.astype('uint8')*255, shape) > 0

        # mask (redundant, since mask is derived from depth image)
        # depth = pre.mask_img(depth, mask)

        # resize
        depth = pre.resize_depth_img(depth, shape)

        # 01 -> 01c
        depth = depth[..., None]

        # preprocess
        depth = pre.preprocess_img(
            depth,
            mask=mask_resized,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True)

        # convert to correct data format
        if K.image_data_format() == 'channels_last':
            axes = 'b01c'
        else:
            axes = 'bc01'
        depth = img_utils.dimshuffle(depth, '01c', axes)

        # repeat if sampling is enabled
        if args.n_samples > 1:
            depth = np.repeat(depth, args.n_samples, axis=0)

    # prepare rgb input
    if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        # load
        rgb = img_utils.load(rgb_filepath)

        # create mask
        # if args.input_type == INPUT_RGB:
        #     # derive mask from rgb image
        #     mask = rgb > 0
        #     mask_resized = pre.resize_mask(mask.astype('uint8')*255,
        #                                    shape) > 0
        # else:
        #     # mask rgb image using mask derived from depth image
        #    rgb = pre.mask_img(rgb, mask)

        # resize
        rgb = pre.resize_depth_img(rgb, shape)

        # preprocess
        rgb = pre.preprocess_img(
            rgb,
            mask=mask_resized,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True)

        # convert to correct data format
        if K.image_data_format() == 'channels_last':
            axes = 'b01c'
        else:
            axes = 'bc01'
        rgb = img_utils.dimshuffle(rgb, '01c', axes)

        # repeat if sampling is enabled
        if args.n_samples > 1:
            rgb = np.repeat(rgb, args.n_samples, axis=0)

    # return preprocessed images
    if args.input_type == INPUT_DEPTH_AND_RGB:
        return depth, rgb
    if args.input_type == INPUT_DEPTH:
        return depth
    if args.input_type == INPUT_RGB:
        return rgb


def main(model_name,
         weights_filepath,
         image_or_image_basepath,
         input_type=INPUT_DEPTH,
         input_width=96,
         input_height=96,
         input_preprocessing='standardize',
         n_samples=1,
         output_type=OUTPUT_BITERNION,
         n_classes=8,
         mobilenet_v2_alpha=1.0,
         devices='0',
         cpu=False,
         verbose=False):

    # use cpu or gpu
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    # print settings
    print("Settings")
    print("------------")
    print(f"model:               {model_name}")
    print(f"weights_filepath:    {weights_filepath}")
    print(f"image_or_image_basepath:    {image_or_image_basepath}")
    print(f"input_type:          {input_type}")
    print(f"input_width:         {input_width}")
    print(f"input_height:        {input_height}")
    print(f"input_preprocessing: {input_preprocessing}")
    print(f"n_samples:           {n_samples}")
    print(f"output_type:         {output_type}")
    print(f"n_classes:           {n_classes}")
    print(f"mobilenet_v2_alpha:  {mobilenet_v2_alpha}")
    print(f"devices:             {devices}")
    print(f"cpu only:            {cpu}")
    print(f"verbose:             {verbose}")
    print()

    # load network ------------------------------------------------------------
    model = load_network(model_name, weights_filepath,
                         input_type, input_height, input_width,
                         output_type,
                         sampling=n_samples > 1,
                         n_classes=n_classes,
                         mobilenet_v2_alpha=mobilenet_v2_alpha)

    # print some information about model
    if verbose:
        print(f"Model summary")
        print("------------")
        print(model.summary())

    # load images -------------------------------------------------------------
    # get input files
    if input_type == INPUT_DEPTH_AND_RGB:
        input_extensions = ('.png', '.jpg', '.pgm')
        input_files = get_files_by_extension(
            image_or_image_basepath,
            extension=input_extensions)
        # create tuple of files
        input_files = [(f, f.replace('_depth.png', '.jpg'),
                        f.replace('_depth.png', '_mask.png'))
                       for f in input_files if '_depth.png' in f]
    elif input_type == INPUT_DEPTH:
        input_extensions = ('.png', '.pgm')
        input_files = get_files_by_extension(
            image_or_image_basepath,
            extension=input_extensions)
        # create tuple of files
        input_files = [(f, f.replace('.png', '_mask.png'))
                       for f in input_files]
    else:
        input_extensions = ('.jpg', '.png')
        input_files = get_files_by_extension(
            image_or_image_basepath,
            extension=input_extensions)
        # create tuple of files
        input_files = [(f, f.replace('.jpg', '_mask.png')) for f in input_files]

    # when single image was given, input_files contains a single tuple
    n_images = len(input_files)

    print(f"Found {n_images} images to process.")
    print()

    # create output folder if directory was given
    if os.path.isdir(image_or_image_basepath):
        outpath = os.path.join(image_or_image_basepath, 'results')
        if not os.path.exists(outpath):
            os.makedirs(outpath)

    # process images ----------------------------------------------------------
    results = np.zeros((n_images, n_samples))
    times = np.zeros(n_images)
    times_smoothing = np.zeros(n_images)
    for i, files in enumerate(input_files):
        # load and preprocess images
        images = load_and_preprocess(files, args)

        # inference
        if verbose:
            print(f"Process image: {files[0]} ({i+1}/{n_images})")
            print("-------------")
        start = time.time()
        pred = model.predict(images)
        duration = time.time() - start

        if verbose:
            print(f"Time: {duration*1000:5.1f} ms")
            print()

        # get image name (name without extension)
        image_name = os.path.splitext(os.path.split(files[0])[1])[0]

        # postprocess
        if n_samples == 1:
            results[i] = post.postprocess_prediction(pred,
                                                     output_type=output_type,
                                                     as_deg=False)
        else:
            # n_samples > 1: use dropout sampling
            angles = np.zeros((n_samples,))
            for j in range(n_samples):
                angles[j] = post.postprocess_prediction(
                    pred[j, ...],
                    output_type=output_type,
                    as_deg=False)

            mean_angle = circmean(angles, high=np.pi, low=-np.pi)
            std_angle = circstd(angles, high=np.pi, low=-np.pi)

            results[i] = mean_angle

            if verbose:
                print(f"STD angle: {std_angle*180/np.pi:3.1f} deg")

            # save plot
            outname = os.path.join(outpath, f"{image_name}_samples.png")
            _plot_sampling_results(angles, mean_angle, std_angle, outname)

        times[i] = duration

    # print timing ------------------------------------------------------------
    print("Timing")
    print("------------")
    print(f"mean:   {np.mean(times)*1000:5.1f} ms")
    print(f"median: {np.median(times)*1000:5.1f} ms")
    print()

    # return results
    return results


def _plot_sampling_results(samples, mean, std, filename):
    """Plot sampling results"""
    samples = samples * 180 / np.pi
    mean = mean * 180 / np.pi
    std = std * 180 / np.pi

    # create histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(samples, density=True, bins=180//5)

    # plot mean
    ax.axvline(mean, color='k')

    # plot std
    x = np.arange(-180, 181, 0.1)
    ax.plot(x, norm.pdf(x, mean, std), color='k', linestyle='--')

    ax.set_xlim([-180, 180])

    ax.set_xlabel("Angle in deg")
    ax.set_ylabel("Density")

    # save
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def ros_node():
    """
    Main ROS node that sets up the service and handles incoming requests.
    """
    rospy.init_node('orientation_estimator')  # Initialize the ROS node
    rospy.Service('/pose/estimate', Trigger, handle_estimation_request)  # Create service

    rospy.loginfo("Orientation Estimator Node Initialized. Waiting for requests...")
    rospy.spin()  # Keep the node running


def handle_estimation_request(req):
    """
    Handle the incoming service request to process orientation estimation.
    """
    # Parse the command-line arguments
    global args
    args = _parse_args()

    # Perform orientation estimation
    rospy.loginfo("Received request for orientation estimation")
    results = main(args.model,
                   args.weights_filepath,
                   args.image_or_image_basepath,
                   args.input_type,
                   args.input_width,
                   args.input_height,
                   args.input_preprocessing,
                   args.n_samples,
                   args.output_type,
                   args.n_classes,
                   args.mobilenet_v2_alpha,
                   args.devices,
                   args.cpu,
                   args.verbose)

    rospy.loginfo(f"Orientation estimation completed. Results: {results}")

    # Call the /pose/continue service to signal continuation
    try:
        rospy.wait_for_service('/pose/continue')
        continue_service = rospy.ServiceProxy('/pose/continue', Empty)
        continue_service()
        rospy.loginfo("Called /pose/continue service successfully.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to /pose/continue failed: {e}")

    return TriggerResponse(
        success=True,
        message="Orientation estimation and continuation completed successfully."
    )


if __name__ == "__main__":
    # Run the ROS node
    ros_node()
