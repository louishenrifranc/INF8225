language: PYTHON
name:   "myTP2"

variable {
    name: "batch_size"
    type: INT
    size: 1
    min: 1
    max: 11369
}

variable {
    name: "optimizer"
    type: ENUM
    size: 1
    options: "sgd"
    options: "momemtum"
    options: "nesterov"
    options: "rmsprop"
}

variable {
    name: "patience"
    type: INT
    size: 1
    min: 5
    max: 50
}

variable {
    name: "learning_rate"
    type: FLOAT
    size: 1
    min: 0.000001
    max: 0.01
}

variable {
    name: "gaussian_noise"
    type: ENUM
    size: 1
    options: "True"
    options: "False"
}

variable {
    name: "momentum"
    type: FLOAT
    size: 1
    min: 0.5
    max: 1.3
}
