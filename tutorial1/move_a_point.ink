schema GameState
    # These names (and types) have to match the dictionary returned by get_state() our simulator.
    Float32 current_x,
    Float32 current_y,
    Float32 target_x,
    Float32 target_y
end

schema PlayerMove
    # This name (and type) has to match the parameter to advance() in
    # our simulator. We specify the range and step size for the action.
    Float32{0:1.575:6.283} direction_radians
end

schema SimConfig
    # The sim doesn't have any configuration, but Inkling requires
    # that we have to have a schema anyway (for now).
    Int8 dummy
end

# This will be our only concept -- trying to get to the target
concept find_the_target
    is classifier      # We're picking one of a few options
    predicts (PlayerMove)
    follows input(GameState)
    feeds output
end

# This is the Inkling name of our simulator. It has to match the parameter
# passed to the PointSimulator() constructor, but does not have to match the
# name of the Python file.
simulator move_a_point_sim(SimConfig)
    action (PlayerMove)
    state (GameState)
end

# we can name the curriculum anything we want
curriculum learn_curriculum
    # this is the name of our concept from above
    train find_the_target
    # this is our simulator name
    with simulator move_a_point_sim
    # This is the name of our objective function in the simulator
    objective reward_closeness
        lesson get_close 
            configure
                constrain dummy with Int8{-1}
            until
                # This is again the name of the objective function
                maximize reward_closeness
end
