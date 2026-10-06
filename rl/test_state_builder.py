from state_builder import StateBuilder



detection_output = {


    "attack_probability":0.94,

    "failed_requests":35,

    "requests_5m":120,

    "hour":22,

    "is_night":0,

    "privileged":1,

    "critical_resource":1,

    "has_certificate":0,

    "user_risk":0.87

}



state = StateBuilder.build(

    detection_output

)



print("STATE:")

print(state)



print()

print(
    "VALID:",
    StateBuilder.validate(state)
)