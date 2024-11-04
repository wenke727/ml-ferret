import sys
sys.path.append('../ferretui')

from mm_utils import get_anyres_image_grid_shape


def test_get_anyres_image_grid_shape():
    # Setup: Provide the inputs based on debug information
    image_sizes = [(1170, 2532)]
    image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    image_size = 336  # Obtained from self.get_vision_tower().config.image_size

    # Expected outputs
    expected_num_patch_width = 1
    expected_num_patch_height = 3

    # Call the function
    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_sizes[0], image_grid_pinpoints, image_size
    )

    # Assertions to check if function returns expected values
    assert num_patch_width == expected_num_patch_width, f"Expected num_patch_width to be {expected_num_patch_width}, but got {num_patch_width}"
    assert num_patch_height == expected_num_patch_height, f"Expected num_patch_height to be {expected_num_patch_height}, but got {num_patch_height}"

    print("Test passed!")

# Run the test
test_get_anyres_image_grid_shape()