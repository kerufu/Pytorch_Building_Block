import torch
import numpy as np

np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)

color_channel = 3
input_range = (0, 1)


def get_probe_size(model):
    reverse_receptive_field_shape = (0, 0)
    output_shape = (0, 0)
    probe_size = 256
    while output_shape[0] <= reverse_receptive_field_shape[0] * 2 or output_shape[1] <= reverse_receptive_field_shape[1] * 2:
        probe_size *= 2

        base = np.ones((1, color_channel, probe_size, probe_size)) * input_range[0]
        base = torch.from_numpy(base).float()
        output_base = model(base)
        output_probe = get_probe_output(model, probe_size, (0, 0))

        rrfr = get_reverse_receptive_field_range(output_base, output_probe)

        output_shape = (output_base.shape[2], output_base.shape[3])
        reverse_receptive_field_shape = (rrfr[1]-rrfr[0], rrfr[3]-rrfr[2])
    
    return probe_size, rrfr, output_base

def get_probe_output(model, probe_size, probe_offset):

    probe = np.ones((1, color_channel, probe_size, probe_size)) * input_range[0]
    probe[:, :, probe_size//2+probe_offset[0], probe_size//2+probe_offset[1]] = input_range[1]

    probe = torch.from_numpy(probe).float()

    output_probe = model(probe)

    return output_probe

def get_reverse_receptive_field_range(output_base, output_probe):
    diff = output_probe - output_base
    diff = np.add.reduce(diff.detach().numpy(), axis=(0, 1))
    x_range, y_range = np.where(diff!=0)
    return (min(x_range), max(x_range), min(y_range), max(y_range))

def meansure(model):
    probe_size, rrfr_base, output_base = get_probe_size(model)
    print("probe_size:" ,probe_size)

    x_max = rrfr_base[1]
    y_max = rrfr_base[3]

    for i in range(1, probe_size//2):
        try:
            output_probe = get_probe_output(model, probe_size, (i, 0))
            rrfr = get_reverse_receptive_field_range(output_base, output_probe)
            if rrfr[0] > x_max:
                receptive_field_x_range = i
                break
        except:
            # probe missing due to conincedence between dilation and stride
            pass

    for i in range(1, probe_size//2):
        try:
            output_probe = get_probe_output(model, probe_size, (0, i))
            rrfr = get_reverse_receptive_field_range(output_base, output_probe)
            if rrfr[2] > y_max:
                receptive_field_y_range = i
                break
        except:
            # probe missing due to conincedence between dilation and stride
            pass


    receptive_field_shape = (receptive_field_x_range, receptive_field_y_range)
    print("receptive_field_shape: ", receptive_field_shape)
        






    
