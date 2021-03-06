from dataloaders.datasets import pascal
import torch.utils.data as data

def make_data_loader(args, **kwargs):
    if args.dataset == "pascal":
        train_set = pascal.VOCSegmentation(args, split="train")
        val_set = pascal.VOCSegmentation(args, split="val")

        num_class = train_set.NUM_CLASS
        train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
    else:
        return NotImplementedError

    return train_loader, val_loader, test_loader, num_class
