"""
CS5800 Summer 2024
Final Project
Author: Calvin Lo

This program matches features between two images.
"""

from kd_tree import get_descriptors, custom_match_features, ratio_test
import cv2


def main():
    '''
    Function -- main
        Demonstrates the use of the k-d tree matching functions in a feature
        recognition pipeline.

    Parameters:
        None.

    Returns:
        None.
    '''
    image1_path = 'Image_set_01\\test.jpg'
    image2_path = 'Image_set_01\\reference.jpg'

    keypoints, descriptors = get_descriptors(image1_path)
    query_keypoints, query_descriptors = get_descriptors(image2_path)
    k = len(query_descriptors[0])
    matches = custom_match_features(descriptors,
                                    query_descriptors,
                                    k,
                                    method='bbf')

    filter_ratio = 0.4
    good_matches = ratio_test(matches, filter_ratio)
    print(f"Good matches: {len(good_matches)}")

    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Draw matches
    result_image = \
        cv2.drawMatches(image2,
                        query_keypoints,
                        image1, keypoints,
                        good_matches,
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Scale image
    screen_width = 1280
    screen_height = 1080

    # Get the dimensions of the image
    height, width = result_image.shape[:2]

    # Calculate the scaling factor
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)

    # Resize the image with the scaling factor
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(result_image, (new_width, new_height))

    # Display the resized image
    cv2.imshow('Feature Matching', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
