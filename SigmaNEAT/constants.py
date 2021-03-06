import numba.types as typ

# Neat Data
NEATDATA__INNOVATION_NUMBER_INDEX = 0
NEATDATA__NODE_GENES_INDEX = 1
NEATDATA__CONNECTION_GENES_INDEX = 2
NEATDATA__INPUT_SIZE_INDEX = 3
NEATDATA__OUTPUT_SIZE_INDEX = 4


# Connection Info
CONNECTION_INFO__INPUT_INDEX = 0
CONNECTION_INFO__OUTPUT_INDEX = 1
CONNECTION_INFO__WEIGHT_INDEX = 2
CONNECTION_INFO__ACTIVATIONFUNCTION_INDEX = 3
CONNECTION_INFO__ENABLED_INDEX = 4
CONNECTION_INFO__INNOVATION_NO_INDEX = 5

# Data Types
OPTIONAL_FLOAT = typ.optional(typ.float64).type
