"""
Configuration format:
configs = {
    <data_config_name> : {
        'train' : [
            (<category>,    <split>,    <pose_ratio>,   <no_pose_ratio>),
            ...
        ],
        'val' : [
            (<category>,    <split>),
            ...
        ]
    }
}
"""
configs = {

    ####################################
    # Fewshot pretrain; Multi category #
    ####################################
    "few_shot_train_pose1.0" : {
        "train" :  [
            ("airplanes",   "train",    1.0,    1.0),
            ("cars",        "train",    1.0,    1.0),
            ("chairs",      "train",    1.0,    1.0),
            ("displays",    "train",    1.0,    1.0),
            ("phones",      "train",    1.0,    1.0),
            ("speakers",    "train",    1.0,    1.0),
            ("tables",      "train",    1.0,    1.0),
        ],
        "val" :  [
            ("airplanes",   "val"),
            ("cars",        "val"),
            ("chairs",      "val"),
            ("displays",    "val"),
            ("phones",      "val"),
            ("speakers",    "val"),
            ("tables",      "val"),
        ],
        "test" :  [
            # Training classes
            ("airplanes",   "test"),
            ("cars",        "test"),
            ("chairs",      "test"),
            ("displays",    "test"),
            ("phones",      "test"),
            ("speakers",    "test"),
            ("tables",      "test"),
            # Novel class
            ("benches",     "test"),
            ("cabinets",    "test"),
            ("vessels",     "test"),
        ],
        "e_lr": 1e-3,
        "d_lr": 1e-4,
        "g_lr": 1e-4,
        "decay_steps": 5000,
        "decay_rate": 1.0,
        "validation_interval": 1000,
        "num_batches" : 80000,
        "batch_size" : 64,
    },

    "few_shot_train_pose0.5" : {
        "train" :  [
            ("airplanes",   "train",    0.5,    1.0),
            ("cars",        "train",    0.5,    1.0),
            ("chairs",      "train",    0.5,    1.0),
            ("displays",    "train",    0.5,    1.0),
            ("phones",      "train",    0.5,    1.0),
            ("speakers",    "train",    0.5,    1.0),
            ("tables",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("airplanes",   "val"),
            ("cars",        "val"),
            ("chairs",      "val"),
            ("displays",    "val"),
            ("phones",      "val"),
            ("speakers",    "val"),
            ("tables",      "val"),
        ],
        "test" :  [
            # Training classes
            ("airplanes",   "test"),
            ("cars",        "test"),
            ("chairs",      "test"),
            ("displays",    "test"),
            ("phones",      "test"),
            ("speakers",    "test"),
            ("tables",      "test"),
            # Novel class
            ("benches",     "test"),
            ("cabinets",    "test"),
            ("vessels",     "test"),
        ],
        "e_lr": 1e-3,
        "d_lr": 1e-4,
        "g_lr": 1e-4,
        "decay_steps": 5000,
        "decay_rate": 1.0,
        "validation_interval": 1000,
        "num_batches" : 40000,
        "batch_size" : 64,
    },

    "few_shot_train_pose0.1" : {
        "train" :  [
            ("airplanes",   "train",    0.1,    1.0),
            ("cars",        "train",    0.1,    1.0),
            ("chairs",      "train",    0.1,    1.0),
            ("displays",    "train",    0.1,    1.0),
            ("phones",      "train",    0.1,    1.0),
            ("speakers",    "train",    0.1,    1.0),
            ("tables",      "train",    0.1,    1.0),
        ],
        "val" :  [
            ("airplanes",   "val"),
            ("cars",        "val"),
            ("chairs",      "val"),
            ("displays",    "val"),
            ("phones",      "val"),
            ("speakers",    "val"),
            ("tables",      "val"),
        ],
        "test" :  [
            # Training classes
            ("airplanes",   "test"),
            ("cars",        "test"),
            ("chairs",      "test"),
            ("displays",    "test"),
            ("phones",      "test"),
            ("speakers",    "test"),
            ("tables",      "test"),
            # Novel class
            ("benches",     "test"),
            ("cabinets",    "test"),
            ("vessels",     "test"),
        ],
        "e_lr": 1e-3,
        "d_lr": 1e-4,
        "g_lr": 1e-4,
        "decay_steps": 5000,
        "decay_rate": 1.0,
        "validation_interval": 500,
        "num_batches" : 20000,
        "batch_size" : 64,
    },

    "few_shot_train_pose0.01" : {
        "train" :  [
            ("airplanes",   "train",    0.01,    1.0),
            ("cars",        "train",    0.01,    1.0),
            ("chairs",      "train",    0.01,    1.0),
            ("displays",    "train",    0.01,    1.0),
            ("phones",      "train",    0.01,    1.0),
            ("speakers",    "train",    0.01,    1.0),
            ("tables",      "train",    0.01,    1.0),
        ],
        "val" :  [
            ("airplanes",   "val"),
            ("cars",        "val"),
            ("chairs",      "val"),
            ("displays",    "val"),
            ("phones",      "val"),
            ("speakers",    "val"),
            ("tables",      "val"),
        ],
        "test" :  [
            # Training classes
            ("airplanes",   "test"),
            ("cars",        "test"),
            ("chairs",      "test"),
            ("displays",    "test"),
            ("phones",      "test"),
            ("speakers",    "test"),
            ("tables",      "test"),
            # Novel class
            ("benches",     "test"),
            ("cabinets",    "test"),
            ("vessels",     "test"),
        ],
        "e_lr": 1e-3,
        "d_lr": 1e-4,
        "g_lr": 1e-4,
        "decay_steps": 5000,
        "decay_rate": 1.0,
        "validation_interval": 500,
        "num_batches" : 10000,
        "batch_size" : 64,
    },

    ###############################
    # Single Category Experiments #
    ###############################
    "airplanes_pose0.5" : {
        "train" :  [
            ("airplanes",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("airplanes",      "val"),
        ],
        "test" :  [
            ("airplanes",      "test"),
        ],
        "num_batches" : 20000,
    },

    "benches_pose0.5" : {
        "train" :  [
            ("benches",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("benches",      "val"),
        ],
        "test" :  [
            ("benches",      "test"),
        ],
        "num_batches" : 20000,
    },

    "chairs_pose0.5" : {
        "train" :  [
            ("chairs",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("chairs",      "val"),
        ],
        "test" :  [
            ("chairs",      "test"),
        ],
        "num_batches" : 20000,
    },

    "cars_pose0.5" : {
        "train" :  [
            ("cars",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("cars",      "val"),
        ],
        "test" :  [
            ("cars",      "test"),
        ],
        "num_batches" : 20000,
    },

    "sofas_pose0.5" : {
        "train" :  [
            ("sofas",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("sofas",      "val"),
        ],
        "test" :  [
            ("sofas",      "test"),
        ],
        "num_batches" : 20000,
    },

    "tables_pose0.5" : {
        "train" :  [
            ("tables",      "train",    0.5,    1.0),
        ],
        "val" :  [
            ("tables",      "val"),
        ],
        "test" :  [
            ("tables",      "test"),
        ],
        "num_batches" : 20000,
    },
}
