import numpy as np
import tensorflow as tf
from tqdm import tqdm
from astropy.io import fits
import glob
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_center(img, cropx, cropy):

    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    #print(startx)
    starty = y // 2 - (cropy // 2)
    #print(starty)
    return img[starty:starty + cropy, startx:startx + cropx, ...]

def augment_dataset(images, image_names, num_augmented_images=20):
    augmented_images = []
    augmented_image_names = []
    if images.ndim == 3:
        images = images[..., np.newaxis]
    for i, image in tqdm(enumerate(images)):
        augmented_images.append(image)
        augmented_image_names.append(image_names[i])
        # Perform data augmentation multiple times to create more augmented images
        for _ in range(num_augmented_images):

            # Random rotation (angle in radians)
            angle = tf.random.uniform(shape=[], minval=-0.5*np.pi, maxval=0.5*np.pi)
            augmented_image = tf.image.rot90(image, k=tf.cast(angle / (0.5*np.pi), tf.int32))

            augmented_images.append(augmented_image)
            augmented_image_names.append(image_names[i])

    augmented_image_names = np.asarray(augmented_image_names)
    augmented_images = np.array(augmented_images)

    return augmented_images, augmented_image_names
    
def load_images(filepath, label, crop_center_fn, augment_fn, num_augmented_images):
    images = []
    labels = []
    image_names = []
    logging.info(f"Loading images from {filepath} with label {label}...")
    for image_file in tqdm(glob.glob(filepath + "*.fits")):
        image_names.append(image_file)
        try:
            with fits.open(image_file, memmap=False) as hdul:
                if 'star' in label:
                    img = hdul[0].data
                elif 'dual' in label:
                    img = hdul[0].data
                elif 'offset' in label:
                    img = hdul[0].data
                else:
                    img = hdul[1].data
                img = crop_center_fn(img, 94, 94)
                images.append(img)
                if not augment_fn:
                    labels.append(label)
        except OSError:
            print(f"{image_file} is corrupted or empty, moving on...")
            continue
    images = np.asarray(images)
    if augment_fn and num_augmented_images:
        images, image_names = augment_fn(images, image_names, num_augmented_images=num_augmented_images)
    labels = [label] * len(images)  # Ensure labels are appended correctly for augmented images
    labels = np.asarray(labels)
    print(f"Loaded {len(images)} images with {len(labels)} labels from {filepath}")
    return images, labels, image_names
def create_dataset(empty_sky_filepath = None, single_image_filepath = None, dual_image_filepath = None, offset_image_filepath = None, stellar_filepath = None, train = 0.65, val = 0.2, test = 0.15): #Remember to change original code to account for this change!
    all_images = []
    all_labels = []
    all_filepaths = []
    if empty_sky_filepath:
        empty_sky_images, empty_sky_labels, empty_sky_names = load_images(empty_sky_filepath, "empty_sky", crop_center, augment_dataset, 3)
        all_images.append(empty_sky_images)
        all_labels.append(empty_sky_labels)
        all_filepaths.append(empty_sky_names)
    if single_image_filepath:
        single_images, single_labels, single_image_names = load_images(single_image_filepath, "single_AGN", crop_center, augment_dataset, 5)
        print(f"Length of single AGN images: {len(single_images)}")
        all_images.append(single_images)
        all_labels.append(single_labels)
        all_filepaths.append(single_image_names)
    if dual_image_filepath:
        dual_images, dual_labels, dual_image_names = load_images(dual_image_filepath, "dual_AGN", crop_center, None, None)
        logging.info("expanding dims")
        dual_images = np.expand_dims(dual_images, axis = -1)
        all_images.append(dual_images)
        all_labels.append(dual_labels)
        all_filepaths.append(dual_image_names)
        
    if offset_image_filepath:
        offset_images, offset_labels, offset_image_names = load_images(offset_image_filepath, "offset_AGN", crop_center, None, None)
        logging.info("expanding dims")
        offset_images = np.expand_dims(offset_images, axis = -1)
        all_images.append(offset_images)
        all_labels.append(offset_labels)
        all_filepaths.append(offset_image_names)
        
    if stellar_filepath:
        stellar_images, stellar_labels, stellar_image_names, = load_images(stellar_filepath, "star_AGN_align", crop_center, None, None)
        print(f"Length of stellar images: {stellar_images.shape}")
        logging.info("expanding dims")
        stellar_images = np.expand_dims(stellar_images, axis = -1)
        all_images.append(stellar_images)
        all_labels.append(stellar_labels)
        all_filepaths.append(stellar_image_names)
    
    total_images = np.concatenate(all_images, axis = 0)
    #print(np.shape(all_labels))
    total_labels = np.concatenate(all_labels, axis = 0)
    total_filepaths = np.concatenate(all_filepaths, axis = 0)
    print(f"Total images: {np.shape(total_images)}")
    print(f"Total labels: {np.shape(total_labels)}")
    print(f"Total filepaths: {np.shape(total_filepaths)}")
    #np.random.seed(42)
    indices = np.arange(total_images.shape[0])
    np.random.shuffle(indices)

    shuffled_images = total_images[indices]
    shuffled_labels = total_labels[indices]
    shuffled_filepaths = total_filepaths[indices]

    train_size = int(train * total_images.shape[0])
    val_size = int(val * total_images.shape[0])
    test_size = total_images.shape[0] - train_size - val_size

    train_images = shuffled_images[:train_size]
    train_labels = shuffled_labels[:train_size]
    train_filepaths = shuffled_filepaths[:train_size]

    val_images = shuffled_images[train_size:train_size + val_size]
    val_labels = shuffled_labels[train_size:train_size + val_size]
    val_filepaths = shuffled_filepaths[train_size:train_size + val_size]

    test_images = shuffled_images[train_size + val_size:]
    test_labels = shuffled_labels[train_size + val_size:]
    test_filepaths = shuffled_filepaths[train_size + val_size:]

    print(f"Train_dataset: {train_images.shape}")
    print(f"Train_labels: {train_labels.shape}")
    print(f"Train_filepaths: {train_filepaths.shape}")
    print(f"Val_dataset: {val_images.shape}")
    print(f"Val_labels: {val_labels.shape}")
    print(f"Val_filepaths: {val_filepaths.shape}")
    print(f"Test_dataset: {test_images.shape}")
    print(f"Test_labels: {test_labels.shape}")
    print(f"Test_filepaths: {test_filepaths.shape}")

    return (train_images, train_labels, train_filepaths), (val_images, val_labels, val_filepaths), (test_images, test_labels, test_filepaths)

def make_datasets_other_bands(fltr, model = 'B'):
    flux_ratio_prefix = np.round(np.arange(0.1, 1.0, 0.1), 1)
    arcsec_prefix = np.round(np.arange(0.4, 2.5, 0.05), 2)
    composite_images = []
    for flux_ratio in tqdm(flux_ratio_prefix):
        for asec in arcsec_prefix:
            fits_filepath = f"HSC_survey_bands/{fltr}/{flux_ratio}_flux_ratio/composite_dual_AGN/{asec}_arcsecond_separations"
            for fits_file in glob.glob(fits_filepath+"/*.fits"):
                try:
                    with fits.open(fits_file, memmap = False) as hdu1:
                        img = hdu1[0].data
                        img = crop_center(img, 60, 60)
                        img = np.expand_dims(img, axis = -1)
                        composite_images.append(img)
                except Exception as UnopenFileException:
                    print(f"Could not open file {fits_file} because of an issue (bad/corrupted data)")
    single_images = []
    for north_files in glob.glob(f"HSC_survey_bands/entire_bands/{fltr}/northern_sky/confirmed_single_AGN_north/*.fits"):
         with fits.open(north_files, memmap = False) as hdu2:
            img = hdu2[1].data
            img = crop_center(img, 60, 60)
            img = np.expand_dims(img, axis = -1)
            single_images.append(img)
    for fall_files in glob.glob(f"HSC_survey_bands/entire_bands/{fltr}/fall_equatorial/confirmed_signle_AGN_fall/*.fits"):
         with fits.open(fall_files, memmap = False) as hdu3:
            img = hdu3[1].data
            img = crop_center(img, 60, 60)
            img = np.expand_dims(img, axis = -1)
            single_images.append(img)
    for spring_files in glob.glob(f"HSC_survey_bands/entire_bands/{fltr}/spring_equatorial/confirmed_signle_AGN_spring/*.fits"):
         with fits.open(spring_files, memmap = False) as hdu4:
            img = hdu4[1].data
            img = crop_center(img, 60, 60)
            img = np.expand_dims(img, axis = -1)
            single_images.append(img)
    single_images = np.asarray([arr.reshape((60, 60, 1)) for arr in single_images])
    composite_images = np.asarray([arr.reshape((60, 60, 1)) for arr in composite_images])
    augmented_single_images = augment_dataset(single_images)
    nothing_images = []
    for nothing_files in glob.glob(f"HSC_survey_bands/entire_bands/{fltr}/empty_space_images/*.fits"):
        with fits.open(nothing_files, memmap = False) as hdu5:
            img = hdu5[1].data
            img = crop_center(img, 60, 60)
            img = np.expand_dims(img, axis = -1)
            nothing_images.append(img)
    print(np.shape(composite_images))
    #augmented_single_images = np.expand_dims(augmented_single_images, axis = -1)
    print(np.shape(augmented_single_images))
    #augmented_single_images = np.expand_dims(augmented_single_images, axis = -1)
    #composite_images = np.expand_dims(composite_images, axis = -1)
    
    all_single_labels = []
    for _ in range(len(augmented_single_images)):
        all_single_labels.append("single AGN")
    
    all_double_labels = []
    for _ in range(len(composite_images)):
        all_double_labels.append("double AGN")
    all_nothing_labels = []
    for _ in range(len(nothing_images)):
        all_nothing_labels.append("nothing/indeterminate")
    
   
    modelB_training_data_2 = np.concatenate((augmented_single_images, composite_images, nothing_images), axis = 0)
    
    train_ratio = 0.75
    #make evaluation dataset
    
    modelB_labels_real = []
    for _ in range(len(augmented_single_images)):
        modelB_labels_real.append("single AGN")
    for _ in range(len(composite_images)):
        modelB_labels_real.append("double AGN")
    for _ in range(len(nothing_images)):
        modelB_labels_real.append("nothing/indeterminate")
    modelB_labels_real = np.asarray(modelB_labels_real)
    
    indicesB2 = np.random.permutation(len(modelB_training_data_2))
    shuffled_modelB_2_dataset = modelB_training_data_2[indicesB2]
    shuffled_modelB_labels_real = modelB_labels_real[indicesB2]

    split_index_modelB2 = int(train_ratio * len(shuffled_modelB_2_dataset))
    train_dataset_modelB_2 = shuffled_modelB_2_dataset[:split_index_modelB2]
    train_labels_modelB_2 = shuffled_modelB_labels_real[:split_index_modelB2]
    validation_dataset_modelB_2 = shuffled_modelB_2_dataset[split_index_modelB2:]
    validation_labels_modelB_2 = shuffled_modelB_labels_real[split_index_modelB2:]
    
    
    print(f"Shape of real model B training data: {np.shape(train_dataset_modelB_2)}")
    print(f"Shape of real model B training labels: {np.shape(train_labels_modelB_2)}")
    print(f"Shape of real model B validation data: {np.shape(validation_dataset_modelB_2)}")
    print(f"Shape of real model B validation labels: {np.shape(validation_labels_modelB_2)}")
    return (train_dataset_modelB_2, train_labels_modelB_2), (validation_dataset_modelB_2, validation_labels_modelB_2)
