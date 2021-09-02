import os
from random import random, randint

# Log a parameter (key-value pair)

log_param("param1", randint(0, 100))

# Log a metric; metrics can be updated throughout the run
log_metric("foo", random())
log_metric("foo", random() + 1)
log_metric("foo", random() + 2)

# Log an artifact (output file)
if not os.path.exists("outputs"):
    os.makedirs("outputs")
with open("outputs/test.txt", "w") as f:
    f.write("hello world!")
log_artifacts("outputs")

