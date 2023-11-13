import cv2
import glob
import os

def combine_masks(mask_image_path_1, mask_image_path_2):
  """Combines two masks into a single mask.

  Args:
    mask_image_path_1: The path to the first mask image.
    mask_image_path_2: The path to the second mask image.

  Returns:
    A grayscale mask image that is the union of the two input masks.
  """

  mask_image_1 = cv2.imread(mask_image_path_1, cv2.IMREAD_GRAYSCALE)
  mask_image_2 = cv2.imread(mask_image_path_2, cv2.IMREAD_GRAYSCALE)

  combined_mask = cv2.bitwise_or(mask_image_1, mask_image_2)
  return combined_mask

def replace_masks(original_mask_image_path, combined_mask_image):
  """Replaces two masks with a combined mask.

  Args:
    original_mask_image_path: The path to the original mask image.
    combined_mask_image: The combined mask image.
  """

  cv2.imwrite(original_mask_image_path, combined_mask_image)

# Get the list of mask paths.
mask_image_paths = glob.glob("Dataset_BUSI_with_GT/*/*_mask_1.png")

# Create a list of first mask paths.
first_mask_paths = [mask.replace('_1', '') for mask in mask_image_paths]

# Combine the two masks for each pair of mask paths.
for mask_image_path_1, mask_image_path_2 in zip(mask_image_paths, first_mask_paths):
  combined_mask = combine_masks(mask_image_path_1, mask_image_path_2)

  # Replace the two masks with the combined mask.
  replace_masks(mask_image_path_1, combined_mask)
  os.remove(mask_image_path_1)
