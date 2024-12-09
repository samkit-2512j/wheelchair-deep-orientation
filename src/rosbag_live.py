#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
.. codeauthor:: OpenAI's GPT-4

"""
import os
import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from scipy.stats import norm, circmean, circstd
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

# Import the deep orientation module
from deep_orientation import beyer, mobilenet_v2, beyer_mod_relu
from deep_orientation.inputs import INPUT_TYPES, INPUT_DEPTH, INPUT_RGB, INPUT_DEPTH_AND_RGB
from deep_orientation.outputs import OUTPUT_TYPES, OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION, OUTPUT_BITERNION
import deep_orientation.preprocessing as pre
import deep_orientation.postprocessing as post

import utils.img as img_utils
from utils.io import get_files_by_extension

# Initialize Seaborn
seaborn.set_context('notebook', font_scale=1.2)

# Global variables
model = None
bridge = CvBridge()  # For converting ROS Image messages to OpenCV format
args = None  # Placeholder for parsed arguments

def _parse_args():
    """Parse command-line arguments"""
    desc = 'Apply neural network for orientation estimation using ROS topic images'
    parser = ap.ArgumentParser(description=desc, formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('model',
                        type=str,
                        help=("Model to use: beyer, beyer_mod_relu, or mobilenet_v2"),
                        choices=['beyer', 'beyer_mod_relu', 'mobilenet_v2'])

    parser.add_argument('weights_filepath',
                        type=str,
                        help="Path to the weights to load")

    # ROS parameters -----------------------------------------------------------
    parser.add_argument('ros_topic',
                        type=str,
                        help="Name of the ROS topic to subscribe to for image frames")

    # Input parameters ---------------------------------------------------------
    parser.add_argument('-it', '--input_type',
                        type=str,
                        default=INPUT_DEPTH,
                        choices=INPUT_TYPES,
                        help=(f"Input type. One of {INPUT_TYPES}, default: {INPUT_DEPTH}"))

    parser.add_argument('-iw', '--input_width',
                        type=int,
                        default=96,
                        help="Patch width to use, default: 96")

    parser.add_argument('-ih', '--input_height',
                        type=int,
                        default=96,
                        help="Patch height to use, default: 96")

    parser.add_argument('-ip', '--input_preprocessing',
                        type=str,
                        default='standardize',
                        choices=['standardize', 'scale01', 'none'],
                        help="Preprocessing to apply. One of [standardize, scale01, none], default: standardize")

    # Output parameters --------------------------------------------------------
    parser.add_argument('-n', '--n_samples',
                        type=int,
                        default=1,
                        help="If `n_samples` > 1, dropout sampling is applied, default: 1")

    parser.add_argument('-ot', '--output_type',
                        type=str,
                        default=OUTPUT_BITERNION,
                        choices=OUTPUT_TYPES,
                        help=(f"Output type. One of {OUTPUT_TYPES}, default: {OUTPUT_BITERNION}"))

    parser.add_argument('-nc', '--n_classes',
                        type=int,
                        default=8,
                        help=(f"Number of classes when output_type is {OUTPUT_CLASSIFICATION}, default: 8"))

    # Other parameters ---------------------------------------------------------
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

def load_network(model_name, weights_filepath, input_type, input_height, input_width, output_type, sampling=False, **kwargs):
    """Load the neural network model with specified parameters."""
    # Load the correct model based on the name
    model_module = globals()[model_name]
    model_kwargs = {}

    if model_name == 'mobilenet_v2' and 'mobilenet_v2_alpha' in kwargs:
        model_kwargs['alpha'] = kwargs.get('mobilenet_v2_alpha')
    if output_type == OUTPUT_CLASSIFICATION:
        assert 'n_classes' in kwargs
        model_kwargs['n_classes'] = kwargs.get('n_classes')

    # Retrieve and return the model
    model = model_module.get_model(
        input_type=input_type,
        input_shape=(input_height, input_width),
        output_type=output_type,
        sampling=sampling,
        **model_kwargs
    )

    # Load model weights
    model.load_weights(weights_filepath)

    return model

def preprocess_image(image):
    """Preprocess the image for model input."""
    # Convert the ROS Image message to OpenCV format
    try:
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")
        return None

    # Prepare mask
    mask_resized = cv2.resize(cv_image, (args.input_width, args.input_height))
    mask_resized = mask_resized > 0

    # Prepare the input depending on input type
    if args.input_type in [INPUT_DEPTH, INPUT_DEPTH_AND_RGB]:
        depth = cv_image if args.input_type == INPUT_DEPTH else cv_image[:, :, 0]  # Assuming depth is the first channel
        depth = pre.resize_depth_img(depth, (args.input_height, args.input_width))
        depth = depth[..., None]  # Add channel dimension

        # Preprocess depth image
        depth = pre.preprocess_img(
            depth,
            mask=mask_resized,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True
        )

        # Adjust to correct data format
        if K.image_data_format() == 'channels_last':
            depth = np.expand_dims(depth, axis=-1)
        else:
            depth = np.expand_dims(depth, axis=0)

        if args.n_samples > 1:
            depth = np.repeat(depth, args.n_samples, axis=0)

    if args.input_type in [INPUT_RGB, INPUT_DEPTH_AND_RGB]:
        rgb = cv_image if args.input_type == INPUT_RGB else cv_image[:, :, 1:]  # Assuming RGB is channels 1, 2, 3
        rgb = pre.resize_depth_img(rgb, (args.input_height, args.input_width))

        # Preprocess RGB image
        rgb = pre.preprocess_img(
            rgb,
            mask=mask_resized,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True
        )

        # Adjust to correct data format
        if K.image_data_format() == 'channels_last':
            rgb = np.expand_dims(rgb, axis=-1)
        else:
            rgb = np.expand_dims(rgb, axis=0)

        if args.n_samples > 1:
            rgb = np.repeat(rgb, args.n_samples, axis=0)

    # Return the preprocessed image
    if args.input_type == INPUT_DEPTH_AND_RGB:
        return depth, rgb
    elif args.input_type == INPUT_DEPTH:
        return depth,
    else:
        return rgb,

def postprocess_output(output):
    """Convert the model output to a readable angle."""
    if args.output_type == OUTPUT_BITERNION:
        return post.biternion2deg(output)
    elif args.output_type == OUTPUT_REGRESSION:
        return post.rad2deg(output)
    else:
        return post.class2deg(np.argmax(output, axis=-1), args.n_classes)

def visualize_results(nw_inputs, output):
    """Visualize the model's inputs and the predicted orientation."""
    plt.clf()  # Clear the current figure
    fig = plt.figure(figsize=(8, 6))
    
    # Visualize inputs
    for j, inp in enumerate(nw_inputs):
        img = inp[0]

        if K.image_data_format() == 'channels_last':
            axes = '01c'
        else:
            axes = 'c01'
        img = img_utils.dimshuffle(img, axes, '01c')

        img = pre.preprocess_img_inverse(
            img,
            scale01=args.input_preprocessing == 'scale01',
            standardize=args.input_preprocessing == 'standardize',
            zero_mean=True,
            unit_variance=True
        )

        ax = fig.add_subplot(1, len(nw_inputs) + 1, j + 1)
        if img.shape[-1] == 1:
            ax.imshow(img[:, :, 0], cmap='gray', vmin=img[img != 0].min(), vmax=img.max())
        else:
            ax.imshow(img)
        ax.axis('off')

    # Visualize output
    ax = fig.add_subplot(1, len(nw_inputs) + 1, len(nw_inputs) + 1, polar=True)
    ax.set_theta_zero_location('S', offset=0)
    ax.hist(np.deg2rad(output), width=np.deg2rad(2), density=True, alpha=0.5 if args.n_samples > 1 else 1.0, color='#1f77b4')

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

def image_callback(msg):
    """Callback function for ROS Image message."""
    # Preprocess the image
    nw_inputs = preprocess_image(msg)

    if nw_inputs is None:
        return

    # Predict using the model
    nw_output = model.predict(nw_inputs, batch_size=args.n_samples)

    # Postprocess the output
    output = postprocess_output(nw_output)

    # Visualize the results
    visualize_results(nw_inputs, output)

def main():
    """Main function to set up ROS node and subscribe to the topic."""
    global model, args

    # Parse command-line arguments
    args = _parse_args()

    # Set up device configuration
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        args.devices = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    if not args.devices or args.model == 'mobilenet_v2':
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')

    # Load the neural network model
    model = load_network(
        args.model,
        args.weights_filepath,
        args.input_type,
        args.input_height,
        args.input_width,
        args.output_type,
        sampling=args.n_samples > 1,
        n_classes=args.n_classes,
        mobilenet_v2_alpha=args.mobilenet_v2_alpha
    )

    # Initialize ROS node
    rospy.init_node('orientation_estimator', anonymous=True)

    # Subscribe to the ROS image topic
    rospy.Subscriber(args.ros_topic, Image, image_callback)

    # Spin to keep the script running and receiving callbacks
    rospy.spin()

if __name__ == '__main__':
    main()
