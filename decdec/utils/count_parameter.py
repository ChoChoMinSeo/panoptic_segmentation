def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def cal_size(num_params,dtype):
    if dtype == "float32":
        return (num_params/1024**2) * 4
    elif dtype == "float16":
        return (num_params/1024**2) * 2
    elif dtype == "int8":
        return (num_params/1024**2) * 1
    else:
        return -1
def model_analysis(model):
    print('# Trainable Parameters:',format(count_trainable_parameters(model),','))
    print('# Parameters:',format(count_parameters(model),','))
    print('Model Size {:.2f} MB'.format(cal_size(count_trainable_parameters(model),'float32')))