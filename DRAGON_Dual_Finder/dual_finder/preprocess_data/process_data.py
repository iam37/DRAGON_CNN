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

def augment_dataset(images, num_augmented_images=20):
    augmented_images = []
    if images.ndim == 3:
        images = images[..., np.newaxis]
    for i, image in tqdm(enumerate(images)):
        augmented_images.append(image)

        # Perform data augmentation multiple times to create more augmented images
        for _ in range(num_augmented_images):

            # Random rotation (angle in radians)
            angle = tf.random.uniform(shape=[], minval=-0.5*np.pi, maxval=0.5*np.pi)
            augmented_image = tf.image.rot90(image, k=tf.cast(angle / (0.5*np.pi), tf.int32))

            augmented_images.append(augmented_image)

    augmented_images = np.array(augmented_images)

    return augmented_images

def create_dataset(empty_sky_filepath, single_image_filepath, dual_image_filepath, train = 0.65, val = 0.2, test = 0.15): #Remember to change original code to account for this change!
    empty_sky_images = []
    empty_sky_labels = []
    logging.info("Loading empty sky images...")
    for empty_sky_files in tqdm(glob.glob(empty_sky_filepath+"*.fits")):
        with fits.open(empty_sky_files, memmap = False) as hdul:
            img = hdul[1].data
            img = crop_center(img, 94, 94)
            empty_sky_images.append(img)
            #empty_sky_labels.append("empty_sky")
    #print(np.shape(empty_sky_images))
    empty_sky_images = np.asarray(empty_sky_images)
    empty_sky_images = augment_dataset(empty_sky_images, num_augmented_images = 3)
    for _ in empty_sky_images:
        empty_sky_labels.append("empty_sky")
    #empty_sky_images = empty_sky_images[:, :, :, -1]
    empty_sky_labels = np.asarray(empty_sky_labels)
    print(f"Shape of empty_sky_images: {empty_sky_images.shape}")
    if len(empty_sky_images) != len(empty_sky_labels):
        print(f"ERROR: empty sky images has length {len(empty_sky_images)} while empty sky labels has length {len(empty_sky_labels)}")

    logging.info("Loading single AGN images...")
    single_images = []
    single_labels = []
    for single_image in tqdm(glob.glob(single_image_filepath+"*.fits")):
        with fits.open(single_image) as hdul:
            img = hdul[1].data
            img = crop_center(img, 94, 94)
            single_images.append(img)
            #single_labels.append("single_AGN")
    single_images = np.asarray(single_images)
    single_images = augment_dataset(single_images, num_augmented_images = 10)
    for _ in single_images:
        single_labels.append("single_AGN")
    #single_images = single_images[:, :, :, -1]
    single_labels = np.asarray(single_labels)
    if len(single_images) != len(single_labels):
        print(f"ERROR: single images has length {len(single_images)} while single labels has length {len(single_labels)}")


    logging.info("Loading dual AGN images...")
    dual_images = []
    dual_labels = []
    for dual_image in glob.glob(dual_image_filepath+"*.fits"):
        try:
            with fits.open(dual_image) as hdul:
                img = hdul[0].data
                img = crop_center(img, 94, 94)
                dual_images.append(img)
                dual_labels.append("dual_AGN")
        except OSError:
            print(f"{dual_image} is corrupted or empty, moving on...")
            continue
    #dual_images = augment_dataset(dual_images, num_augmented_images = 5)
    #dual_images = dual_images[:, :, :, -1]
    dual_images = np.expand_dims(dual_images, axis = -1)
    dual_images = np.asarray(dual_images)
    dual_labels = np.asarray(dual_labels)
    if len(single_images) != len(single_labels):
        print(f"ERROR: single images has length {len(single_images)} while single labels has length {len(single_labels)}")

    # Now make splits of the dataset into training, testing, and validation metrics. The defaults for this function are train = 0.65, val = 0.2, test = 0.15

    total_images = np.concatenate((empty_sky_images, single_images, dual_images), axis = 0)
    total_labels = np.concatenate((empty_sky_labels, single_labels, dual_labels), axis = 0)
    print(f"Total images: {np.shape(total_images)}")
    print(f"Total labels: {np.shape(total_labels)}")

    np.random.seed(42)
    indices = np.arange(total_images.shape[0])
    np.random.shuffle(indices)

    shuffled_images = total_images[indices]
    shuffled_labels = total_labels[indices]

    train_size = int(train * total_images.shape[0])
    val_size = int(val * total_images.shape[0])
    test_size = total_images.shape[0] - train_size - val_size

    train_images = shuffled_images[:train_size]
    train_labels = shuffled_labels[:train_size]

    val_images = shuffled_images[train_size:train_size + val_size]
    val_labels = shuffled_labels[train_size:train_size + val_size]

    test_images = shuffled_images[train_size + val_size:]
    test_labels = shuffled_labels[train_size + val_size:]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

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
