def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan
def normalize(image_arr):
    mean = np.mean(image_arr)
    std = np.std(image_arr)
    normalized_arr = (image_arr - mean) / std
    return normalized_arr
def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
def apply_n4_bias_field_correction(img):
    img = ants.from_numpy(img)
    mask = ants.get_mask(img)
    img = ants.n4_bias_field_correction(img,mask= mask, rescale_intensities=True, shrink_factor=2, convergence={'iters': [50, 50, 30, 20], 'tol': 1e-07}, spline_param=None, return_bias_field=False, verbose=False, weight_mask=None)
    img = img.numpy()
    return img

def process_scan(path):
    volume = read_nifti_file(path)
    normalized_volume = normalize(volume)
    resized_volume = resize_volume(normalized_volume)
    bias_corrected_volume = apply_n4_bias_field_correction(resized_volume)
    return bias_corrected_volume
