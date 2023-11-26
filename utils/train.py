import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import keras
import tensorflow as tf
from utils.augmentation import get_preprocessing
from utils.data import Dataloder, Dataset

import models.segmentation_models_qubvel as sm

from wandb.keras import WandbMetricsLogger


def run_train(pref, X_train_ir, y_train, X_test_ir, y_test, train_config, model, model_arch, use_wandb=False, augmentation=None, class_weights=None, training_mode='fine_tune'):
    """
    Training function.

    Parameters:
    -----------
        pref : str
            identifier for training run
        X_train : numpy.ndarray
            train images
        y_train : numpy.ndarray
            image labels
        X_test : numpy.ndarray
            test images
        y_test : numpy.ndarray
            test labels
        train_config : dict
            stores num_epochs, loss, backbone, optimizer, batch_size, learning_rate
        model : keras.engine.functional.Functional
            model defined before call
        model_arch : str
            model architecture 
        use_wandb : Bool
            whether to use wandb for train monitoring
        augmentation : albumentations.core.composition.Compose
            specifies on-fly augmentation methods (if to be appplied; else None)
        class_weights : list
            the class weights to use (None when no weights should be used)
        fold_no : int
            if in hyperparameter optimization, number of the crossfold run
        training_mode : str
            either 'fine_tune' or 'hyperparameter_tune'

    Return:
    ------
        scores, hist_val_iou   
            generalization metrics and history of generalization metrics
    """
    
    CLASSES=['Background', 'Water', 'Building_No_Damage', 'Building_Minor_Damage', 'Building_Major_Damage', 'Building_Total_Destruction', 
            'Vehicle', 'Road-Clear', 'Road-Blocked', 'Tree', 'Pool']
    WEIGHTS = class_weights
    
    # training dataset
    train_dataset = Dataset(
        images_ir=X_train_ir, 
        masks=y_train, 
        classes=CLASSES, 
        augmentation=augmentation,
        preprocessing=get_preprocessing(sm.get_preprocessing(train_config['backbone'])),
    )

    # validation dataset
    valid_dataset = Dataset(
        images_ir=X_test_ir,
        masks=y_test, 
        classes=CLASSES,
        preprocessing=get_preprocessing(sm.get_preprocessing(train_config['backbone'])),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # define loss
    if train_config['loss'] == 'jaccard':
        LOSS = sm.losses.JaccardLoss(class_weights=class_weights)
    elif train_config['loss'] == 'focal_dice':
        dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
        focal_loss = sm.losses.CategoricalFocalLoss()
        LOSS = dice_loss + (1 * focal_loss)
    elif train_config['loss'] == 'categoricalCE':
        LOSS = sm.losses.CategoricalCELoss(class_weights=class_weights)
    elif train_config['loss']== 'focal':
        LOSS = sm.losses.CategoricalFocalLoss()
    else:
        print('No loss function specified')

    # define optimizer
    if train_config['optimizer'] == 'Adam':
        OPTIMIZER = keras.optimizers.Adam()
    elif train_config['optimizer'] == 'SGD':
        OPTIMIZER = keras.optimizer.SGD()
    elif train_config['optimizer'] == 'Adamax':
        OPTIMIZER = keras.optimizer.Adamax()
    else:
        print('No optimizer specified')

    # define evaluation metrics
    mean_iou = sm.metrics.IOUScore(name='mean_iou')
    weighted_iou = sm.metrics.IOUScore(class_weights=class_weights, name='weighted_iou')
    f1 = sm.metrics.FScore(beta=1, name='f1')
    precision = sm.metrics.Precision(name='precision')
    recall = sm.metrics.Recall(name='recall')
    background_iou = sm.metrics.IOUScore(class_indexes=0, name='background_iou')
    water_iou = sm.metrics.IOUScore(class_indexes=1, name='water_iou')
    building_no_damage_iou = sm.metrics.IOUScore(class_indexes=2, name='building_no_damage_iou')
    building_minor_damage_iou = sm.metrics.IOUScore(class_indexes=3, name='building_minor_damage_iou')
    building_major_damage_iou = sm.metrics.IOUScore(class_indexes=4, name='building_major_damage_iou')
    building_total_des_iou = sm.metrics.IOUScore(class_indexes=5, name='building_total_des_iou')
    vehicle_iou = sm.metrics.IOUScore(class_indexes=6, name='vehicle_iou')
    road_clear_iou = sm.metrics.IOUScore(class_indexes=7, name='road_clear_iou')
    road_blocked_iou = sm.metrics.IOUScore(class_indexes=8, name='road_blocked_iou')
    tree_iou = sm.metrics.IOUScore(class_indexes=9, name='tree_iou')
    pool_iou = sm.metrics.IOUScore(class_indexes=10, name='pool_iou')
    rounded_iou = sm.metrics.IOUScore(threshold=0.5, name='mean_iou_rounded')

    # compile model
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[mean_iou, weighted_iou, f1, precision, recall, background_iou,
                                                           water_iou, building_no_damage_iou, building_no_damage_iou,
                                                           building_minor_damage_iou, building_major_damage_iou, 
                                                           building_total_des_iou, vehicle_iou, road_clear_iou, 
                                                           road_blocked_iou, tree_iou, pool_iou, rounded_iou])

    # define callbacks
    if training_mode == 'hyperparameter_tune':
        callbacks = [
            tf.keras.callbacks.CSVLogger('./metrics/scores/{0}/{1}.csv'.format(training_mode, pref)),
        ]

    # when in fine-tuning, save weights of best performing model in terms of minimal val_loss
    else:
        weights_path = './weights/{}/'.format(model_arch)
        os.makedirs(weights_path, exist_ok = True)
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(weights_path, 'best_model{1}.h5'.format(model_arch, pref)), save_weights_only=True, save_best_only=True, mode='min'),
            tf.keras.callbacks.CSVLogger('./metrics/scores/{0}/{1}.csv'.format(training_mode, pref)),
        ]

    if use_wandb:
        callbacks.append(WandbMetricsLogger())


    history = model.fit(train_dataloader,
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=len(train_dataloader), 
                        epochs=train_config['num_epochs'],  
                        validation_data=valid_dataloader, 
                        validation_steps=len(valid_dataloader),
                        shuffle=False)

    # generalization metrics of trained model
    scores = model.evaluate(valid_dataloader, verbose=0)
    hist_val_iou = history.history['val_mean_iou']
        
    return scores, hist_val_iou
