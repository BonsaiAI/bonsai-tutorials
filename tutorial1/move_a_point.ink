schema GameState
    # EXERCISE: Create a name (and type) for each variable of x and y position.
    # These have to match the dictionary returned by get_state() our simulator.
    # <Your code goes here>
end

schema PlayerMove
    # EXERCISE: This name (and type) has to match the parameter to step() in
    # our simulator. You need to specify the range and step size for the action.
    # <Your code goes here>
end

schema SimConfig
    # The simulator doesn't have any configuration, but Inkling requires
    # that we have a schema anyway.
    Int8 dummy
end

# This will be our only concept -- trying to get to the target
concept find_the_target
    is classifier
    predicts (PlayerMove)
    follows input(GameState)
    feeds output
end

# This is the Inkling name of our simulator. It has to match the parameter
# to bonsai.run_for_training_or_prediction(), but does not have to match the
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
    objective reward_shaped
        lesson get_close 
            configure
                constrain dummy with Int8{-1}
            until
                # This is again the name of the objective function
                maximize reward_shaped
end
