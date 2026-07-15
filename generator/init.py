from faker import Faker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from ..config import (
    DATA_DIR, GENERATED_DATA_SIZE, RANDOM_SEED,
    PAP_MEDICATIONS, PAP_ELIGIBILITY_CRITERIA
)

fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)