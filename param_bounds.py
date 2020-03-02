def match(x1, x2):
    if x1 == x2:
        return True
    elif set(x1) == set(x2):
        pos = x2.index(x1[0])
        rearr_x2 = x2[pos:] + x2[1:pos] + [x2[pos]]
        return x1 == rearr_x2
    else:
        return False


def param_bounds(neurons, ecs, n_num, param, delta, iterations):

    pstart = neurons[n_num].get_ref(param)
    result = [pstart, pstart]
    neurons, ecs, time, potentials, activations, u_rates, tact_dur, rhythm = calculate(neurons, ecs, iterations)
    flag = 0
    for i in range(1, 100):
        neurons[n_num].param_change(param, pstart + abs(pstart) * i)
        neurons, ecs, time, potentials, activations, u_rates, tact_dur, rhythm1 = calculate(neurons, ecs, iterations)

        if not match(rhythm, rhythm1):
            flag = 1
            break
    pb_right = neurons[n_num].get_ref(param)

    pb_left = pstart
    print(pb_left, pb_right)
    if flag == 1:
        while abs(pb_right - pb_left) > delta:
            print(neurons[n_num].get_ref(param), pb_right, pb_left)

            neurons[n_num].param_change(param, (pb_left + pb_right) / 2)
            neurons, ecs, time, potentials, activations, u_rates, tact_dur, rhythm1 = calculate(neurons, ecs, iterations)
            if not match(rhythm, rhythm1):
                pb_right = neurons[n_num].get_ref(param)
            else:
                pb_left = neurons[n_num].get_ref(param)

    result[1] = pb_left
    neurons[n_num].param_change(param, pstart)

    flag = 0
    for i in range(1, 100):
        neurons[n_num].param_change(param, pstart + abs(pstart) * (-i))
        neurons, ecs, time, potentials, activations, u_rates, tact_dur, rhythm1 = calculate(neurons, ecs, iterations)

        if not match(rhythm, rhythm1):
            flag = 1
            print(neurons[n_num].get_ref(param))
            break
    pb_right = pstart
    pb_left = neurons[n_num].get_ref(param)
    print(pb_left, pb_right)
    if flag == 1:
        while abs(pb_right - pb_left) > delta:
            neurons[n_num].param_change(param, pb_right - abs(pb_left - pb_right) / 2)
            neurons, ecs, time, potentials, activations, u_rates, tact_dur, rhythm1 = calculate(neurons, ecs, iterations)
            print(neurons[n_num].get_ref(param), pb_right, pb_left)

            if not match(rhythm, rhythm1):
                pb_left = pb_right - abs(pb_left - pb_right) / 2
            else:
                pb_right = pb_right - abs(pb_left - pb_right) / 2
    result[0] = pb_left

    return result