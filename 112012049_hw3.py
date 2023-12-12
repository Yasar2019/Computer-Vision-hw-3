import cv2
import numpy as np
import math
import os


# Q1: Function to apply Gaussian blur to a grayscale image
def gaussian_blur(image, kernel_size, sigma):
    # Create a Gaussian kernel. The kernel size should be odd.
    k = kernel_size // 2
    x, y = np.mgrid[-k : k + 1, -k : k + 1]
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= 2 * np.pi * sigma**2
    # Normalize the kernel so that the sum is 1
    gaussian_kernel /= gaussian_kernel.sum()
    # Apply the Gaussian kernel to the grayscale image
    blurred_image = cv2.filter2D(image, -1, gaussian_kernel)
    return blurred_image


# Q2: Function to perform Canny Edge Detection manually
def canny_edge_detection(image, low_threshold, high_threshold):
    # Compute gradients along the X and Y axis
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180

    # Non-maximum suppression
    nms_image = np.zeros_like(magnitude, dtype=np.uint8)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            q = 255
            r = 255

            # angle 0
            if (0 <= orientation[i, j] < 22.5) or (157.5 <= orientation[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # angle 45
            elif 22.5 <= orientation[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # angle 90
            elif 67.5 <= orientation[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # angle 135
            elif 112.5 <= orientation[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms_image[i, j] = magnitude[i, j]
            else:
                nms_image[i, j] = 0

    # Double threshold
    high_threshold_image = nms_image > high_threshold
    low_threshold_image = (nms_image >= low_threshold) & (nms_image <= high_threshold)

    # Edge Tracking by Hysteresis
    final_image = np.zeros_like(magnitude, dtype=np.uint8)
    # Weak pixels adjacent to strong pixels are considered as strong ones
    for i in range(1, high_threshold_image.shape[0] - 1):
        for j in range(1, high_threshold_image.shape[1] - 1):
            if low_threshold_image[i, j]:
                if (
                    high_threshold_image[i + 1, j - 1]
                    or high_threshold_image[i + 1, j]
                    or high_threshold_image[i + 1, j + 1]
                    or high_threshold_image[i, j - 1]
                    or high_threshold_image[i, j + 1]
                    or high_threshold_image[i - 1, j - 1]
                    or high_threshold_image[i - 1, j]
                    or high_threshold_image[i - 1, j + 1]
                ):
                    final_image[i, j] = 255
            elif high_threshold_image[i, j]:
                final_image[i, j] = 255

    return final_image


# Q3 : Apply Hough Transform
# Function to perform Hough Transform
def hough_transform(image, threshold):
    # Define the Hough space
    thetas = np.deg2rad(np.arange(-90, 90, 1))  # Theta range
    width, height = image.shape
    diag_len = int(np.ceil(np.sqrt(width**2 + height**2)))  # Diagonal length
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)  # Rho range

    # Initialize the accumulator
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)

    # Find edge points (non-zero values)
    y_idxs, x_idxs = np.nonzero(image)

    # Vote in the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx in range(len(thetas)):
            # Calculate rho for each theta
            rho = int(
                (x * np.cos(thetas[theta_idx]))
                + (y * np.sin(thetas[theta_idx]))
                + diag_len
            )
            accumulator[rho, theta_idx] += 1

    # Apply a threshold to the accumulator
    accumulator[accumulator < threshold] = 0

    return accumulator, thetas, rhos


def draw_hough_lines_on_image(image, accumulator, rhos, thetas, threshold):
    # Iterate over the accumulator
    for rho_idx in range(len(rhos)):
        for theta_idx in range(len(thetas)):
            # Check if there is a line at this position
            if accumulator[rho_idx, theta_idx] > threshold:
                # Convert polar coordinates to cartesian coordinates of the line
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Draw the line on the image
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return image


# Pipeline to process the image and save the results
def process_and_save_images(image_path, output_dir):
    # Read the image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Extract base name without the extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Apply Gaussian Blur
    blurred_image = gaussian_blur(gray_image, kernel_size=5, sigma=1)
    blurred_image_path = os.path.join(output_dir, f"{base_name}_q1.png")
    cv2.imwrite(blurred_image_path, blurred_image)

    # Apply Canny Edge Detection
    canny_edges = canny_edge_detection(
        blurred_image, low_threshold=10, high_threshold=25
    )
    canny_edges_path = os.path.join(output_dir, f"{base_name}_q2.png")
    cv2.imwrite(canny_edges_path, canny_edges)

    # Apply Hough Transform
    hough_acc, thetas, rhos = hough_transform(canny_edges, threshold=120)

    # Draw Hough Lines
    hough_lines_image = draw_hough_lines_on_image(
        original_image, hough_acc, rhos, thetas, threshold=30
    )
    hough_lines_image_path = os.path.join(output_dir, f"{base_name}_q3.png")
    cv2.imwrite(hough_lines_image_path, hough_lines_image)

    return blurred_image_path, canny_edges_path, hough_lines_image_path


def process_images_in_directory(image_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image paths from the directory
    image_paths = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Process each image
    for image_path in image_paths:
        # Process the image and save the results
        (
            blurred_image_file,
            canny_edges_file,
            hough_lines_image_file,
        ) = process_and_save_images(image_path, output_dir)
        print(f"Processed {image_path}")
        print(f"Saved blurred image to {blurred_image_file}")
        print(f"Saved canny edges to {canny_edges_file}")
        print(f"Saved hough lines image to {hough_lines_image_file}")


# Define the directory containing your images and output directory
image_dir = "test_img"
output_dir = "result_img"

# Process all images in the directory
process_images_in_directory(image_dir, output_dir)
