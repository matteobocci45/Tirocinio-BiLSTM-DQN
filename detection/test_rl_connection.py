import numpy as np

from detection.inference import build_detection_output
from rl.state_builder import StateBuilder



probability = 0.94



context = {

    "failed_requests":25,

    "requests_5m":80,

    "hour":23,

    "is_night":1,

    "privileged":1,

    "critical_resource":1,

    "has_certificate":0,

    "user_risk":0.85

}



output = build_detection_output(

    probability,

    context

)



print("Detection output:")
print(output)



state = StateBuilder.build(

    output

)



print("\nRL STATE:")
print(state)