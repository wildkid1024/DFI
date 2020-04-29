
import nn_models.cifar as cifar_models
import nn_models.imagenet as imagenet_models


def load_models(dataset_name='cifar10', model_name='alexnet', **kwargs):
    if dataset_name.startswith('cifar'):
        import nn_models.cifar as models

        model_names = sorted(name for name in models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(models.__dict__[name]))
        
        return cifar_models.__dict__[model_name.lower()](**kwargs)

    elif dataset_name.startswith('imagenet'):
        import torchvision.models as models
        import nn_models.imagenet as customized_models

        default_model_names = sorted(name for name in models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(models.__dict__[name]))

        customized_models_names = sorted(name for name in customized_models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(customized_models.__dict__[name]))
        
        for name in customized_models.__dict__:
            if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
                models.__dict__[name] = customized_models.__dict__[name]

        model_names = default_model_names + customized_models_names
    
    model_name = model_name.lower()

    assert model_name in model_names, 'Model not in the library'

    model = models.__dict__[model_name](**kwargs)

    return model


        


        

